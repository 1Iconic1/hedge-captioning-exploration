# Libraries
import json
import pandas as pd
from openai import OpenAI
import os

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")
pd.set_option('display.max_colwidth', None)

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

def save_results_json(output_path, org_dataset=None, T_atomics=None, g_atomics=None,
                      evaluations=None, metric_name="cap_f1", update_existing=None, limit=2):
    """
    Save image caption + atomic statements + optional evaluation info to a JSON file.
    Use `update_existing` to load from a previous JSON and append evaluation only.

    Parameters:
    - org_dataset: list of dicts from original json
    - T_atomics: list of string results (optional)
    - g_atomics: list of string results (optional)
    - evaluations: list of dicts with TPs, FPs, FNs, Counts (optional)
    - metric_name: the evaluation metric name (e.g., "cap_f1", "BLEU")
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
                        line.strip() for line in model_entry.get("atomic_captions", "").split("\n") if line.strip()
                    ] for model_entry in g_atomics[i]
                }

            result_item = {
                "file_name": item.get("file_name"),
                "human_captions": human_captions,
                "model_captions": item["model_captions"],
                "evaluation": {
                    metric_name: {
                        "T_atomics": [line.strip() for line in T_atomics[i].split("\n") if line.strip()] if T_atomics else [],
                        "g_atomics": model_outputs,
                        "scores": {}
                    }
                }
            }
            results.append(result_item)

    if evaluations:
        for i in range(min(len(results), len(evaluations))):
            metric_scores = {}
            for model_eval in evaluations[i]:
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
            results[i]["evaluation"][metric_name]["scores"] = metric_scores

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved JSON to: {output_path}")


def call_gpt4o(prompt):
    """
    Call GPT 
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=1,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return completion.choices[0].message.content

def parse_atomic_statements(captions):
    """
    Call GPT to convert given captions into atomic statements.
    """
    prompt = (
        "I will give you multiple captions describing the same image.\n"
        "Your task is to:\n"
        "1. Break each caption into fully atomic statements, each expressing exactly one simple and objective fact.\n"
        "2. Each atomic statement must describe only one idea: either object existence, attribute (like color or material), position, or relationship.\n"
        "3. Do not use compound or complex sentences. Avoid words like 'and', 'but', or commas that connect multiple facts.\n"
        "4. Remove any subjective, inferred, or emotional content. Keep only visually verifiable facts.\n"
        "5. Return each atomic statement as a single plain sentence, one per line, without numbering or bullet points.\n\n"
        "Captions:\n"
        + "\n".join(captions)
    )

    return call_gpt4o(prompt)

def remove_duplicate_atomic_statements(captions):
    """
    Call GPT to remove duplicate atomic statements.
    """
    prompt = (
        "You will be given a list of short atomic sentences. Each one expresses a single visual fact from an image.\n"
        "Your task is to identify and remove semantically overlapping or redundant sentences.\n"
        "Only keep one sentence per unique fact, and prefer the clearest or most specific version if two are similar.\n"
        "Do not rewrite or merge sentences. Just delete duplicates.\n"
        "Return the final list as plain text — one sentence per line. No numbering or bullet points.\n\n"
        "Atomic Statements:\n"
        + "\n".join(captions)
    )
    return call_gpt4o(prompt)

def calculate_recall_gpt(T_atomics, g_atomics):
    """
    Call GPT to get TP, FP, FN metrics comparing T and g.
    """
    
    prompt = (
        "You are evaluating the alignment between two sets of atomic statements describing the same image.\n\n"
        "Definitions:\n"
        "- True Positive (TP): A generated atomic statement that correctly matches a human-written atomic statement.\n"
        "- False Negative (FN): A human-written atomic statement that is NOT covered by any generated statement.\n\n"
        "Instructions:\n"
        "1. Compare each generated statement against the human-written statements.\n"
        "2. For each generated statement, mark it as TP based on semantic match.\n"
        "3. For each human-written statement, if no generated statement covers it, mark it as FN.\n\n"
        "Return a single JSON object with:\n"
        "- \"TPs\": a list of all true positive generated statements\n"
        "- \"FNs\": a list of all false negative human-written statements\n"
        "- \"Counts\": a dictionary with the count of TP, FN\n\n"
        "Output must be valid JSON. Do NOT include any markdown formatting like ```json or any extra explanation.\n\n"
        f"Human-written atomic statements:\n{chr(10).join(T_atomics)}\n\n"
        f"Generated atomic statements:\n{chr(10).join(g_atomics)}"
    )

    return call_gpt4o(prompt)

def calculate_precision_gpt(transcript, g_atomics):
    """
    Ask GPT-4o to determine whether each generated atomic statement is
    consistent with the original transcript, for precision evaluation.
    """

    prompt = (
        "You are evaluating the precision of atomic statements generated from an image caption.\n\n"
        "Input:\n"
        "- A raw human written transcript that describes the image.\n"
        "- A list of atomic statements generated by a model.\n\n"
        "Your task:\n"
        "1. For each generated atomic statement, determine if it is consistent with the transcript.\n"
        "2. Label each statement as either:\n"
        "   - True Positive (TP): if it is consistent with the transcript.\n"
        "   - False Positive (FP): if it is not consistent or contradicts the transcript.\n\n"
        "Output:\n"
        "Return a single JSON object with:\n"
        "- \"TPs\": list of true positive generated statements\n"
        "- \"FPs\": list of false positive generated statements\n"
        "- \"Counts\": a dictionary with the count of TP and FP\n\n"
                "Output must be valid JSON. Do NOT include any markdown formatting like ```json or any extra explanation.\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Generated atomic statements:\n{chr(10).join(g_atomics)}"
    )

    return call_gpt4o(prompt)

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

    for item in org_caption[:limit]:
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
        human_result= remove_duplicate_atomic_statements(human_atomic_captions)


        # Model captions
        model_results = []
        for mc in item["model_captions"]:
            model_name = mc["model_name"]
            model_caption = mc["caption"]
            atomic_result = parse_atomic_statements([model_caption])

            model_results.append({
                "model_name": model_name,
                "atomic_captions": atomic_result
            })

        T_atomics.append(human_result)
        g_atomics.append(model_results)

    return T_atomics, g_atomics

def evaluate_matching_file(parsed_dataset):
    eval_outputs = []

    for item in parsed_dataset:
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

    for i in range(len(T_atomics)):
        T_atomics = T_atomics[i]
        g_atomics = g_atomics[i]

        model_outputs = []
        for g_item in g_atomics:
            model_name = g_item["model_name"]
            g_captions = g_item["atomic_captions"]
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

            model_outputs.append({
                "model_name": model_name,
                "recall": recall_parsed_result,
                "precision": precision_parsed_result
            })
    
        eval_outputs.append(model_outputs)

    return eval_outputs

def calculate_cap_f1(evaluation):
    total_output = []
    for item in evaluation:  # item is a list of model evaluations
        model_outputs = []
        for model_scores in item:
            model_name = model_scores["model_name"]

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
