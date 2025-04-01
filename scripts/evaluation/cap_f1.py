# Libraries
import json
import pandas as pd
from openai import OpenAI
import os
import evaluate
from cider.cider import Cider
from tqdm import tqdm

from pydantic import BaseModel
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")
pd.set_option('display.max_colwidth', None)



class AtomicSentences(BaseModel):
    atomic_captions: list[str]

class RecallCounts(BaseModel):
    TP: int
    FN: int

class Recall(BaseModel):
    TPs: list[str]
    FNs: list[str]
    Counts: RecallCounts

class PrecisionCounts(BaseModel):
    TP: int
    FP: int

class Precision(BaseModel):
    TPs: list[str]
    FPs: list[str]
    Counts: PrecisionCounts


def read_json(caption_file, keys):
    """
    Read JSON file and extract only the specified keys.

    Inputs:
    - caption_file: path to JSON file
    - keys: list of keys to extract from each item in the JSON

    Output:
    - list(dictionary)
    """
    with open(caption_file, "r", encoding="utf-8") as f:
        caption_dataset_json = json.load(f)

    parsed_data = []

    for item in caption_dataset_json:
        parsed_item = {}
        for key in keys:
            parsed_item[key] = item[key]
        parsed_data.append(parsed_item)

    return parsed_data

def save_results_json(output_path, org_dataset=None, T_atomics=None, g_atomics=None, metadata=None,
                      evaluations=None, metric_name="cap_f1", update_existing=None, limit=2):
    """
    Save image caption + atomic statements + optional evaluation info to a JSON file.
    Use `update_existing` to load from a previous JSON and append evaluation only.

    Parameters:
    - org_dataset: list of dicts from original json
    - T_atomics: list of string results (optional)
    - g_atomics: list of string results (optional)
    - metadata: list of dicts with TPs, FPs, FNs, Counts (optional)
    - evaluations: list of dicts with recall, precision, cap_f1 or other metrics (optional)
    - metric_name: the evaluation metric name (e.g., "cap_f1", "BLEU", "METEOR", "ROUGE")
    - update_existing: path to existing parsed json if you're only appending evaluation
    """
    results = []

    if update_existing:
        with open(update_existing, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        for i, item in enumerate(org_dataset[:limit] if limit else org_dataset):
            human_captions = [
                hc["caption"] if isinstance(hc, dict) else hc
                for hc in item["human_captions"]
                if (hc["caption"] if isinstance(hc, dict) else hc) != "Quality issues are too severe to recognize visual content."
            ]

            model_outputs = {}
            if g_atomics:
                model_outputs = {
                    model_entry["model_name"]: [
                        line.strip() for line in model_entry.get("atomic_captions", []) if line.strip()
                    ] for model_entry in g_atomics[i]
                }

            result_item = {
                "file_name": item.get("file_name"),
                "human_captions": human_captions,
                "model_captions": item["model_captions"],
                "evaluation": {
                    metric_name: {
                        "T_atomics": [
                            line.strip()
                            for t in T_atomics
                            for line in t.get("atomic_captions", [])
                            if line.strip()
                        ] if T_atomics else [],
                        "g_atomics": model_outputs,
                        "scores": {}
                    }
                }
            }
            results.append(result_item)

    if metadata:
        for i in range(min(len(results), len(metadata))):
            metric_scores = {}
            for model_eval in metadata[i]:
                model_name = model_eval.get("model_name")
                metric_scores[model_name] = {
                    "recall": {
                        "TPs": model_eval.get("recall", {}).get("TPs", []),
                        "FNs": model_eval.get("recall", {}).get("FNs", []),
                        "Counts": model_eval.get("recall", {}).get("Counts", {})
                    },
                    "precision": {
                        "TPs": model_eval.get("precision", {}).get("TPs", []),
                        "FPs": model_eval.get("precision", {}).get("FPs", []),
                        "Counts": model_eval.get("precision", {}).get("Counts", {})
                    }
                }
            results[i]["evaluation"].setdefault(metric_name, {})["metadata"] = metric_scores

    if evaluations and metric_name == "cap_f1":
        for i in range(len(results)):
            metric_scores = results[i]["evaluation"].setdefault(metric_name, {}).setdefault("scores", {})
            for model_eval in evaluations[i]:
                model_name = model_eval.get("model_name")
                metric_scores[model_name] = {k: v for k, v in model_eval.items() if k != "model_name"}
            results[i]["evaluation"][metric_name]["scores"] = metric_scores

    if evaluations and metric_name != "cap_f1":
        for i in range(min(len(results), len(evaluations))):
            eval_list = evaluations[i] if isinstance(evaluations[i], list) else evaluations
            for model_eval in eval_list:
                model_name = model_eval["model_name"]
                for sub_metric in ["BLEU", "METEOR", "ROUGE", "CIDEr"]:
                    results[i].setdefault("evaluation", {}).setdefault(sub_metric, {}).setdefault("scores", {})[model_name] = model_eval.get(sub_metric)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved JSON to: {output_path}")


def call_gpt4o(system_message, user_message, output_format=AtomicSentences):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=1,
        response_format=output_format,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )
    return json.loads(completion.choices[0].message.content)

def parse_atomic_statements(captions):
    """
    Call GPT to convert given captions into atomic statements.
    """
    system_message = (
        "You are an assistant that extracts fully atomic visual facts from image captions designed to output JSON. "
        "Your task is to: "
        "1. Break each caption into fully atomic statements, each expressing exactly one simple and objective fact.\n"
        "2. Each atomic statement must describe only one idea: either object existence, attribute (like color or material), position, or relationship.\n"
        "3. Do not use compound or complex sentences. Avoid words like 'and', 'but', or commas that connect multiple facts.\n"
        "4. Remove any subjective, inferred, or emotional content. Keep only visually verifiable facts.\n"
        "5. Return each atomic statement as a single plain sentence, one per line, without numbering or bullet points.\n\n"
    )

    user_message = (
        "Please convert the following captions into atomic statements.\n"
        "Captions:\n"
        + "\n".join(captions)
    )

    return call_gpt4o(system_message, user_message)


def remove_duplicate_atomic_statements(captions):
    """
    Call GPT to remove duplicate atomic statements using system/user message separation.
    """
    system_message = (
        "You are a helpful assistant that removes semantically redundant or overlapping atomic statements. Design to output JSON. "
        "Each atomic statement expresses a single visual fact from an image. "
        "Only keep one sentence per unique fact, and prefer the clearest or most specific version if two are similar. "
        "Do not rewrite or merge sentences. Just delete duplicates. "
        "Return each atomic statement as a single plain sentence, one per line, without numbering or bullet points."
    )

    user_message = (
        "Atomic Statements:\n"
        + "\n".join(captions)
    )

    return call_gpt4o(system_message, user_message)

def calculate_recall_gpt(T_atomics, g_atomics):
    """
    Call GPT to evaluate recall between human (T_atomics) and generated (g_atomics) atomic statements.
    """
    system_message = (
        "You are an assistant that evaluates the semantic recall of a model's generated atomic statements against a ground truth set of human-written atomic statements.\n\n"

        "Definitions:\n"
        "- True Positive (TP): A human-written atomic statement whose meaning is correctly captured in any of the model's generated atomic statements.\n"
        "- False Negative (FN): A human-written atomic statement that is not reflected in any of the model's generated statements.\n\n"

        "Instructions:\n"
        "1. For each human-written atomic statement, check whether any generated statement captures its meaning.\n"
        "2. If yes, include it in the TP list.\n"
        "3. If not, include it in the FN list.\n"
        "4. Ignore extra information in the generated statements — only evaluate whether each human-written statement is successfully captured.\n"
        "5. Do not assume or infer facts that are not explicitly present in the statements.\n\n"

        "Return a JSON object with:\n"
        "- \"TPs\": list of human-written statements that were matched (true positives).\n"
        "- \"FNs\": list of human-written statements that were missed (false negatives).\n"
        "- \"Counts\": a dictionary with the number of TP and FN statements.\n\n"

        "Only return the JSON object. Do NOT include explanations or any markdown formatting."
    )

    user_message = (
        "Human-written atomic statements:\n" +
        "\n".join(T_atomics) +
        "\n\nGenerated atomic statements:\n" +
        "\n".join(g_atomics)
    )

    return call_gpt4o(system_message, user_message, Recall)

def calculate_precision_gpt(human_captions, g_atomics):
    """
    Call GPT to evaluate the semantic precision between human-written captions and model-generated atomic statements.
    """

    system_message = (
        "You are an assistant that evaluates the **semantic precision** of atomic statements generated by a model based on human-written image captions.\n\n"

        "Definitions:\n"
        "- True Positive (TP): A generated atomic statement that is semantically supported by, or reasonably implied by, at least one human-written caption. Exact wording is not required.\n"
        "- False Positive (FP): A generated atomic statement that introduces information not present in, or contradictory to, any of the human-written captions.\n\n"

        "Instructions:\n"
        "1. Evaluate each generated atomic statement independently.\n"
        "2. If the core meaning of a generated statement is explicitly stated or reasonably implied by any human-written caption, mark it as a True Positive (TP).\n"
        "3. If the statement includes details that are not found or are contradicted by the captions, mark it as a False Positive (FP).\n"
        "4. Accept paraphrased or partially matching statements as TP if the core meaning aligns.\n"
        "5. Do not make assumptions based on common knowledge, visual conventions, or brand familiarity unless explicitly mentioned in the captions.\n\n"

        "Return a JSON object in the following format:\n"
        "- \"TPs\": a list of true positive generated atomic statements.\n"
        "- \"FPs\": a list of false positive generated atomic statements.\n"
        "- \"Counts\": a dictionary with the number of TP and FP statements.\n\n"

        "Only return the JSON object. Do NOT include any explanations or markdown formatting."
    )

    user_message = (
        "Human-written captions:\n" +
        "\n".join(f"- {caption}" for caption in human_captions) +
        "\n\nGenerated atomic statements:\n" +
        "\n".join(f"- {statement}" for statement in g_atomics)
    )

    return call_gpt4o(system_message, user_message, Precision)



def generate_atomic_statement(org_caption, limit=2):
    """
    Generates atomic statements from original human and model captions.

    Inputs:
    - org_caption: list of dicts, each with 'human_captions' and 'model_captions'
    - limit: how many examples to process

    Outputs:
    - T_atomics: list of GPT outputs from human captions
    - g_molmo_atomics: list of GPT outputs from molmo captions
    """
    T_atomics = []
    g_atomics = []

    for item in tqdm(org_caption[:limit]):
        # Filter out human captions
        human_captions = [
            hc["caption"]
            for hc in item["human_captions"]
            if hc["caption"] != "Quality issues are too severe to recognize visual content."
        ]
        human_atomic_captions = []
        for hc in human_captions:
            human_atomic_captions.append(parse_atomic_statements(hc))

        # print(json.dumps(human_atomic_captions, indent=4, ensure_ascii=False))
        total_sentence = [s for output in human_atomic_captions for s in output["atomic_captions"]]
        human_result = remove_duplicate_atomic_statements(total_sentence)


        # Model captions
        model_results = []
        for mc in item["model_captions"]:
            model_name = mc["model_name"]
            model_caption = mc["caption"]
            atomic_result = parse_atomic_statements([model_caption])

            model_results.append({
                "model_name": model_name,
                "atomic_captions": atomic_result["atomic_captions"]
            })

        T_atomics.append(human_result)
        g_atomics.append(model_results)

    return T_atomics, g_atomics

def evaluate_matching_file(parsed_dataset):
    eval_outputs = []

    for item in tqdm(parsed_dataset):
        T_org = item["human_captions"]
        T_atomics = item["evaluation"]["cap_f1"]["T_atomics"]
        g_atomics = item["evaluation"]["cap_f1"]["g_atomics"]

        model_outputs = []
        for g_item in g_atomics:

            model_name = g_item
            g_captions = g_atomics[g_item]

            recall_result = calculate_recall_gpt(T_atomics, g_captions) 
            precision_result = calculate_precision_gpt(T_org, g_captions) 

            try:
                recall_parsed_result = json.loads(recall_result)
                precision_parsed_result = json.loads(precision_result)
            except json.JSONDecodeError:
                print(f"Failed to parse recall result. {recall_result}")
                print(f"Failed to parse precision result. {precision_result}")
                recall_parsed_result = {}
                precision_parsed_result = {}

            # print(json.dumps(recall_parsed_result, indent=4, ensure_ascii=False))
            # print(json.dumps(precision_parsed_result, indent=4, ensure_ascii=False))

            model_outputs.append({
                "model_name": model_name,
                "recall": recall_parsed_result,
                "precision": precision_parsed_result
            })
    
        eval_outputs.append(model_outputs)

    return eval_outputs

def evaluate_matching(T_org, T_atomics, g_atomics):
    eval_outputs = []

    for i in tqdm(range(len(T_atomics))):
        T_atomic = T_atomics[i]
        g_atomic = g_atomics[i]

        model_outputs = []

        for g_item in g_atomic:
            model_name = g_item["model_name"]
            g_captions = g_item["atomic_captions"]

            recall_result = calculate_recall_gpt(T_atomic, g_captions) 
            precision_result = calculate_precision_gpt(T_org[i], g_captions) 

            model_outputs.append({
                "model_name": model_name,
                "recall": recall_result,
                "precision": precision_result
            })
    
        eval_outputs.append(model_outputs)

    return eval_outputs

def calculate_cap_f1(evaluation):
    total_output = []
    for item in tqdm(evaluation):  # item is a list of model evaluations
        model_outputs = []
        for model_scores in item:
            model_name = model_scores["model_name"]
            if "Counts" not in model_scores["precision"] or model_scores["precision"]["Counts"] is None:
                continue

            precision_TP = model_scores["precision"]["Counts"]["TP"]
            precision_FP = model_scores["precision"]["Counts"]["FP"]
            recall_TP = model_scores["recall"]["Counts"]["TP"]
            recall_FN = model_scores["recall"]["Counts"]["FN"]

            precision = precision_TP / (precision_TP + precision_FP) if (precision_TP + precision_FP) != 0 else 0
            recall = recall_TP / (recall_TP + recall_FN) if (recall_TP + recall_FN) != 0 else 0
            cap_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

            # print(f"precision: {precision:.4f}, recall: {recall:.4f}, cap_f1: {cap_f1:.4f}")
            
            model_outputs.append({
                "model_name": model_name,
                "recall": recall,
                "precision": precision,
                "cap_f1": cap_f1
            })

        total_output.append(model_outputs)
    return total_output

def calculate_cap_f1_file(evaluation):
    for item in evaluation:
        for model_name, model_scores in item["evaluation"]["cap_f1"]["scores"].items():
            print(f"Model: {model_name}")
            print(json.dumps(model_scores, indent=4, ensure_ascii=False))

            precision_TP = model_scores["precision"]["Counts"]["TP"]
            precision_FP = model_scores["precision"]["Counts"]["FP"]
            recall_TP = model_scores["recall"]["Counts"]["TP"]
            recall_FN = model_scores["recall"]["Counts"]["FN"]

            precision = precision_TP / (precision_TP + precision_FP) if (precision_TP + precision_FP) != 0 else 0
            recall = recall_TP / (recall_TP + recall_FN) if (recall_TP + recall_FN) != 0 else 0
            cap_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

            print(f"precision: {precision:.4f}, recall: {recall:.4f}, cap_f1: {cap_f1:.4f}")


def get_others(org_caption_dataset, human_captions, limit=2):
    cider_metric = Cider()

    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    eval_outputs = []
    for idx, item in enumerate(tqdm(org_caption_dataset[:limit])):
        references = [human_captions]  # for BLEU, METEOR, ROUGE
        cider_references = {str(idx): human_captions}
        
        for mc in item["model_captions"]:
            model_name = mc["model_name"]
            model_caption = mc["caption"]
            predictions = [model_caption]
            cider_predictions = {str(idx): [model_caption]}
            cider_score, _ = cider_metric.compute_score(cider_references, cider_predictions)

            eval_outputs.append({
                "model_name": model_name,
                "BLEU": bleu.compute(predictions=predictions, references=references),
                "METEOR": meteor.compute(predictions=predictions, references=references),
                "ROUGE": rouge.compute(predictions=predictions, references=references),
                "CIDEr": cider_score
            })
    
    return eval_outputs

def main():
    print("Read caption json file...")
    keys = ["file_name", "human_captions", "model_captions"]
    org_caption_dataset = read_json("test_org.json", keys)

    print("Generating atomic statements...")
    T_atomics, g_atomics  = generate_atomic_statement(org_caption_dataset)
    save_results_json(output_path="parsed_caption.json", org_dataset=org_caption_dataset, T_atomics=T_atomics, g_atomics=g_atomics)

    keys = ["file_name", "human_captions", "T_atomics", "model_captions", "g_atomics"]
    parsed_dataset = read_json("parsed_caption.json", keys)

    evaluation = evaluate_matching(parsed_dataset)
    save_results_json(output_path="final_with_evaluation.json", update_existing="parsed_caption.json", evaluations=evaluation)

    keys = ["file_name", "human_captions", "T_atomics", "model_captions", "g_atomics", "evaluation"]
    eval_dataset = read_json("final_with_evaluation.json", keys)
    calculate_cap_f1(eval_dataset)


if __name__ == "__main__":
    main()
