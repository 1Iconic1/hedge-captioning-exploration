# Libraries
import json
import pandas as pd
from openai import OpenAI
import os

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")
pd.set_option('display.max_colwidth', None)

def read_json_data(caption_file, keys):
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
                      evaluations=None, update_existing=None, limit=2):
    """
    Save image caption + atomic statements + optional evaluation info to a JSON file.
    Use `update_existing` to load from a previous JSON and append evaluation only.
    
    Parameters:
    - org_dataset: list of dicts from original json
    - T_atomics: list of string results (optional)
    - g_molmo_atomics: list of string results (optional)
    - evaluations: list of dicts with TPs, FPs, FNs, Counts (optional)
    - update_existing: path to existing parsed json if you're only appending evaluation
    """
    results = []

    if update_existing:
        # Load existing JSON to append evaluation
        with open(update_existing, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        # From original dataset and atomic results
        for i, item in enumerate(org_dataset[:limit] if limit else org_dataset):
            human_captions = [
                hc["caption"] if isinstance(hc, dict) else hc
                for hc in item["human_captions"]
                if (hc["caption"] if isinstance(hc, dict) else hc) != "Quality issues are too severe to recognize visual content."
            ]

            model_outputs = []
            if g_atomics:
                for model_entry in g_atomics[i]:
                    atomic_string = model_entry.get("atomic_captions", "")
                    model_outputs.append({
                        "model_name": model_entry.get("model_name"),
                        "atomic_captions": [
                            line.strip() for line in atomic_string.split("\n") if line.strip()
                        ]
                    })

            result_item = {
                "file_name": item.get("file_name"),
                "human_captions": human_captions,
                "T_atomics": [line.strip() for line in T_atomics[i].split("\n") if line.strip()] if T_atomics else [],
                "model_captions": item["model_captions"],
                "g_atomics": model_outputs,
            }
            results.append(result_item)

    # Add evaluations if provided
    if evaluations:
        for i in range(min(len(results), len(evaluations))):
            model_evals = []
            for eval_model in evaluations[i]:
                model_evals.append({
                    "model_name": eval_model.get("model_name"),
                    "recall": {
                        "TPs": eval_model.get("recall", {}).get("TPs", []),
                        "FNs": eval_model.get("recall", {}).get("FNs", []),
                        "Counts": eval_model.get("precision", {}).get("Counts", {})
                    },
                    "precision": {
                        "TPs": eval_model.get("precision", {}).get("TPs", []),
                        "FPs": eval_model.get("precision", {}).get("FPs", []),
                        "Counts": eval_model.get("precision", {}).get("Counts", {})
                    },

                })
            results[i]["evaluation"] = model_evals

    # Save final JSON
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
        "I will give you one or more captions from a single image.\n"
        "Your task is to:\n"
        "1. Parse the descriptions into fully atomic statements, each expressing exactly one simple fact.\n"
        "2. Break compound sentences into smaller atomic units.\n"
        "3. Be as granular as possible — separate object existence, color, material, relative position, and relationships.\n"
        "4. Remove duplicates or semantically overlapping content.\n"
        "5. Return the result as a list of sentences, one atomic fact per line, without any numbering or bullet points.\n\n"
        f"Captions:\n" + "\n".join(captions)
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

        human_result = parse_atomic_statements(human_captions)

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

def evaluate_matching(parsed_dataset):
    eval_outputs = []

    for item in parsed_dataset:
        T_org = item["human_captions"]
        T_atomics = item["T_atomics"]
        g_atomics = item["g_atomics"]

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

            # print(json.dumps(recall_parsed_result, indent=4, ensure_ascii=False))
            # print(json.dumps(precision_parsed_result, indent=4, ensure_ascii=False))

            model_outputs.append({
                "model_name": model_name,
                "recall": recall_parsed_result,
                "precision": precision_parsed_result
            })
    
        eval_outputs.append(model_outputs)

    return eval_outputs

def calculate_cap_f1(evaluation):

    for item in evaluation:
        for model in item["evaluation"]:
            
            print(json.dumps(model, indent=4, ensure_ascii=False))
            model_name = model["model_name"]
            evaluation = model["evaluation"]
            TP = evaluation["Counts"]["TP"]
            FP = evaluation["Counts"]["FP"]
            FN = evaluation["Counts"]["FN"]

            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            cap_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0


            print(f"precision: {precision}, recall: {recall}, cap_f1: {cap_f1}")

def main():
    # print("Read caption json file...")
    # keys = ["file_name", "human_captions", "model_captions"]
    # org_caption_dataset = read_json_data("test_org.json", keys)

    # print("Generating atomic statements...")
    # T_atomics, g_atomics  = generate_atomic_statement(org_caption_dataset)
    # save_results_json(output_path="parsed_caption.json", org_dataset=org_caption_dataset, T_atomics=T_atomics, g_atomics=g_atomics)

    keys = ["file_name", "human_captions", "T_atomics", "model_captions", "g_atomics"]
    parsed_dataset = read_json_data("parsed_caption.json", keys)

    evaluation = evaluate_matching(parsed_dataset)
    save_results_json(output_path="final_with_evaluation.json", update_existing="parsed_caption.json", evaluations=evaluation)

    # keys = ["file_name", "human_captions", "T_atomics", "model_captions", "g_atomics", "evaluation"]
    # eval_dataset = read_json_data("final_with_evaluation.json", keys)
    # calculate_cap_f1(eval_dataset)


if __name__ == "__main__":
    main()
