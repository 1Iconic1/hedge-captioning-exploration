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

def calculate_metrics(T_atomics, g_atomics):
    """
    Call GPT to get TP, FP, FN metrics comparing T and g.
    """
    
    prompt = (
        "You are evaluating the alignment between two sets of atomic statements describing the same image.\n\n"
        "Definitions:\n"
        "- True Positive (TP): A generated atomic statement that correctly matches a human-written atomic statement.\n"
        "- False Positive (FP): A generated atomic statement that does NOT match any human-written atomic statement.\n"
        "- False Negative (FN): A human-written atomic statement that is NOT covered by any generated statement.\n\n"
        "Instructions:\n"
        "1. Compare each generated statement against the human-written statements.\n"
        "2. For each generated statement, mark it as TP or FP based on semantic match.\n"
        "3. For each human-written statement, if no generated statement covers it, mark it as FN.\n\n"
        "Return a single JSON object with:\n"
        "- \"TPs\": a list of all true positive generated statements\n"
        "- \"FPs\": a list of all false positive generated statements\n"
        "- \"FNs\": a list of all false negative human-written statements\n"
        "- \"Counts(TP, FP, FN)\": a dictionary with the count of each\n\n"
        "Output must be valid JSON. Do NOT include any markdown formatting like ```json or any extra explanation.\n\n"
        f"Human-written atomic statements:\n{chr(10).join(T_atomics)}\n\n"
        f"Generated atomic statements:\n{chr(10).join(g_atomics)}"
    )

    return call_gpt4o(prompt)

def evaluate_matching(parsed_dataset):
    eval_outputs = []

    for item in parsed_dataset:
        T_atomics = item["T_atomics"]
        molmo_atomics = item["molmo_atomics"]
        # gpt_atomics = item["gpt_atomics"]
        # ll_atomics = item["ll_atomics"]

        molmo_raw_result = calculate_metrics(T_atomics, molmo_atomics)
        # gpt_raw_result = calculate_metrics(T_atomics, gpt_atomics)
        # ll_raw_result = calculate_metrics(T_atomics, ll_atomics)

        try:
            parsed_result = json.loads(molmo_raw_result)
        except json.JSONDecodeError:
            print("Failed to parse evaluation result.")
            parsed_result = {}

        eval_outputs.append(parsed_result)

    return eval_outputs

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
    g_molmo_atomics = []

    for item in org_caption[:limit]:
        human_result = parse_atomic_statements(item["human_captions"])
        molmo_result = parse_atomic_statements([item["model_captions"]])

        T_atomics.append(human_result)
        g_molmo_atomics.append(molmo_result)

    return T_atomics, g_molmo_atomics

def save_results_json(output_path, org_dataset=None, T_atomics=None, g_molmo_atomics=None,
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
        # Fresh build from original dataset and atomic results
        for i, item in enumerate(org_dataset[:limit] if limit else org_dataset):
            result_item = {
                "file_name": item.get("file_name"),
                "human_captions": item["human_captions"],
                "T_atomics": [line.strip() for line in T_atomics[i].split("\n")] if T_atomics else [],
                "model_captions": item["model_captions"],
                "molmo_atomics": [line.strip() for line in g_molmo_atomics[i].split("\n")] if g_molmo_atomics else [],
            }
            results.append(result_item)

    # Add evaluations if provided
    if evaluations:
        for i in range(min(len(results), len(evaluations))):
            results[i]["evaluation"] = {
                "TPs": evaluations[i].get("TPs", []),
                "FPs": evaluations[i].get("FPs", []),
                "FNs": evaluations[i].get("FNs", []),
                "Counts": evaluations[i].get("Counts(TP, FP, FN)", {})
            }

    # Save final JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved JSON to: {output_path}")

def calculate_cap_f1(evaluation):
    for item in evaluation:
        TP = item["evaluation"]["Counts"]["TP"]
        FP = item["evaluation"]["Counts"]["FP"]
        FN = item["evaluation"]["Counts"]["FN"]

        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        cap_f1 = 2*precision*recall / (precision+recall)

        print(f"precision: {precision}, recall: {recall}, cap_f1: {cap_f1}")

def main():
    print("Read caption json file...")
    keys = ["file_name", "human_captions", "model_captions"]
    org_caption_dataset = read_json_data("caption_output.json", keys)

    print("Generating atomic statements...")
    T_atomics, g_molmo_atomics  = generate_atomic_statement(org_caption_dataset)
    save_results_json(output_path="parsed_caption.json", org_dataset=org_caption_dataset, T_atomics=T_atomics, g_molmo_atomics=g_molmo_atomics)

    keys = ["file_name", "human_captions", "T_atomics", "model_captions", "molmo_atomics"]
    parsed_dataset = read_json_data("parsed_caption.json", keys)

    evaluation = evaluate_matching(parsed_dataset)
    save_results_json(output_path="final_with_evaluation.json", update_existing="parsed_caption.json", evaluations=evaluation)

    keys = ["file_name", "human_captions", "T_atomics", "model_captions", "molmo_atomics", "evaluation"]
    eval_dataset = read_json_data("final_with_evaluation.json", keys)
    calculate_cap_f1(eval_dataset)


if __name__ == "__main__":
    main()
