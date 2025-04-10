# Libraries
import json
import pandas as pd
from openai import OpenAI
import os
import evaluate
from collections import Counter

# from cider.cider import Cider
from tqdm import tqdm

from pydantic import BaseModel

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")
pd.set_option("display.max_colwidth", None)


class AtomicSentences(BaseModel):
    atomic_captions: list[str]


class RecallCounts(BaseModel):
    TP: int
    FN: int

class RecallMatchPair(BaseModel):
    T_atomic: str
    g_atomic: str

class Recall(BaseModel):
    TPs: list[str]
    FNs: list[str]
    Match: list[RecallMatchPair]
    Counts: RecallCounts

class PrecisionMatchPair(BaseModel):
    g_atomic: str
    T_org: str

class PrecisionCounts(BaseModel):
    TP: int
    FP: int

class Precision(BaseModel):
    TPs: list[str]
    FPs: list[str]
    Match: list[PrecisionMatchPair]
    Counts: PrecisionCounts


def read_json(caption_file, keys=None):
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

    if keys is None:
        return caption_dataset_json

    parsed_data = []
    for item in caption_dataset_json:
        parsed_item = {key: item[key] for key in keys if key in item}
        parsed_data.append(parsed_item)

    return parsed_data

# pick images to test if needed

import json

def select_data(image_list_file, json_file, output_file, prev = False):
    """
    Read txt file that has selected filenames and filter json items based on that.

    Inputs:
    - image_list_file: path to txt file containing selected filenames (one per line)
    - json_file: path to json file containing dataset (each item must have 'filename' key)
    - output_file: path to save filtered json

    Output:
    - A new JSON file with filtered items
    """
    # Load the selected filenames from txt file
    with open(image_list_file, "r", encoding="utf-8") as f:
        selected_filenames = set(line.strip() for line in f if line.strip())

    # Load the full dataset from json file
    with open(json_file, "r", encoding="utf-8") as f:
        caption_dataset_json = json.load(f)

    # Filter and modify the dataset
    parsed_data = []



    for item in caption_dataset_json:
        if item.get("file_name") in selected_filenames:
            modified_item = item.copy()

            if prev:                
                if "evaluation" in modified_item:
                    eval_data = modified_item["evaluation"]

                    # Remove T_atomics from previous_cap_f1 if it exists
                    if "previous_cap_f1" in eval_data:
                        eval_data["previous_cap_f1"].pop("T_atomics", None)

                    # Merge contents of cap_f1 into previous_cap_f1
                    if "cap_f1" in eval_data:
                        cap_f1_data = eval_data.pop("cap_f1")
                        if "previous_cap_f1" not in eval_data:
                            eval_data["previous_cap_f1"] = {}
                        eval_data["previous_cap_f1"].update(cap_f1_data)

            parsed_data.append(modified_item)


    # Save filtered data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)

    print(f"{len(parsed_data)} data, JSON {output_file} created with {len(parsed_data)} items.")



def save_results_json(
    output_path,
    org_dataset=None,
    parsed_T= None,
    T_atomics=None,
    g_atomics=None,
    T_org=None, 
    metadata=None,
    evaluations=None,
    metric_name="cap_f1",
    update_existing=None,
    limit=None,
):
    """
    Save image caption + atomic statements + optional evaluation info to a JSON file.
    Use `update_existing` to load from a previous JSON and append evaluation only.

    Parameters:
    - org_dataset: list of dicts from original json
    - T_atomics: list of string results (optional)
    - g_atomics: list of string results (optional)
    - parsd_T: list of string results (optional)
    - metadata: list of dicts with TPs, FPs, FNs, Counts (optional)
    - evaluations: list of dicts with recall, precision, cap_f1 or other metrics (optional)
    - metric_name: the evaluation metric name (e.g., "cap_f1", "BLEU", "METEOR", "ROUGE")
    - update_existing: path to existing parsed json if you're only appending evaluation
    - limit: maximum number of items to process (None = process all)
    """
    results = []

    if update_existing:
        with open(update_existing, "r", encoding="utf-8") as f:
            results = json.load(f)
    elif org_dataset:
        results = org_dataset[:limit] if limit else org_dataset

        for i, item in enumerate(results):
            if parsed_T:
                item["evaluation"].setdefault("cap_f1", {})["parsed_atomics"] = [
                    line.strip()
                    for line in parsed_T[i]
                    if line.strip()
                ]

            if T_atomics:
                item["evaluation"].setdefault("cap_f1", {})["T_atomics"] = [
                    line.strip()
                    for line in T_atomics[i].get("atomic_captions", [])
                    if line.strip()
                ]

            if g_atomics:
                model_outputs = {
                    model_entry["model_name"]: [
                        line.strip()
                        for line in model_entry.get("atomic_captions", [])
                        if line.strip()
                    ]
                    for model_entry in g_atomics[i]
                }
                item["evaluation"].setdefault("cap_f1", {})["g_atomics"] = model_outputs
            
            if T_org:
                item["evaluation"].setdefault("cap_f1", {})["T_org"] = [
                    line.strip()
                    for line in T_org[i]
                    if line.strip()
                ]

    if metadata:
        for i in range(min(len(results), len(metadata))):
            metric_scores = {}
            for model_eval in metadata[i]:
                model_name = model_eval.get("model_name")
                metric_scores[model_name] = {
                    "recall": {
                        "TPs": model_eval.get("recall", {}).get("TPs", []),
                        "FNs": model_eval.get("recall", {}).get("FNs", []),
                        "Match": model_eval.get("recall", {}).get("Match", []),
                        "Counts": model_eval.get("recall", {}).get("Counts", {}),
                    },
                    "precision": {
                        "TPs": model_eval.get("precision", {}).get("TPs", []),
                        "FPs": model_eval.get("precision", {}).get("FPs", []),
                        "Match": model_eval.get("precision", {}).get("Match", []),
                        "Counts": model_eval.get("precision", {}).get("Counts", {}),
                    },
                }
            results[i]["evaluation"].setdefault(metric_name, {})["metadata"] = metric_scores

    if evaluations and metric_name == "cap_f1":
        for i in range(min(len(results), len(evaluations))):
            metric_scores = (
                results[i]["evaluation"]
                .setdefault(metric_name, {})
                .setdefault("scores", {})
            )
            for model_eval in evaluations[i]:
                model_name = model_eval.get("model_name")
                metric_scores[model_name] = {
                    k: v for k, v in model_eval.items() if k != "model_name"
                }
            results[i]["evaluation"][metric_name]["scores"] = metric_scores

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved JSON to: {output_path}")



def check_consistency(model_name, T_atomics, g_captions, recall_TP, recall_FN, precision_TP, precision_FP):
    total_recall = recall_TP + recall_FN
    total_precision = precision_TP + precision_FP

    # Recall consistency
    if sorted(total_recall) != sorted(T_atomics):
        print(f"Error: Recall mismatch for model [{model_name}]")
        print(f"length {len(T_atomics)} vs {len(total_recall)}")
        print(f"T atomics:\n{T_atomics}")
        print(f"Recall TPs:\n{recall_TP}")
        print(f"Recall FNs:\n{recall_FN}")

    # Precision consistency
    if sorted(total_precision) != sorted(g_captions):
        print(f"Error: Precision mismatch for model [{model_name}]")
        print(f"length {len(g_captions)} vs {len(total_precision)}")
        print(f"G atomics:\n{g_captions}")
        print(f"Precision TPs:\n{precision_TP}")
        print(f"Precision FPs:\n{precision_FP}")


def call_gpt4o(system_message, user_message, output_format=None, temperature=0.2):
    """
    Calls GPT-4o to generate a response.

    Args:
        system_message (str): The system message to send to GPT-4o.
        user_message (str): The user message to send to GPT-4o.
        temperature (float, optional): The temperature to use for the response. Defaults to 0.2.
        output_format (Pydantic model, optional): The output format to use for the response. Defaults to None. Use this to force a specific output format, like a JSON object.

    Returns:
        str: The response from GPT-4o. Either as an unstructured string (if output_format is None) or a JSON-parsable object (if output_format is a Pydantic model).
    """
    if output_format is None:
        completion = client.beta.chat.completions.create(
            model="gpt-4o-2024-08-06",
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},

            ],
        )

        # # print amount of tokens used
        # print("Prompt tokens:", completion.usage.prompt_tokens)
        # print("Completion tokens:", completion.usage.completion_tokens)
        # print("Total tokens:", completion.usage.total_tokens)

        return completion.choices[0].message.content
    else:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            temperature=temperature,
            response_format=output_format,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )

        # # print amount of tokens used
        # print("Prompt tokens:", completion.usage.prompt_tokens)
        # print("Completion tokens:", completion.usage.completion_tokens)
        # print("Total tokens:", completion.usage.total_tokens)

        return json.loads(completion.choices[0].message.content)


def call_gpt4o_assist(messages, output_format=None, temperature=0.2):
    """
    Calls GPT-4o to generate a response using a full list of messages.

    Args:
        messages (list): A list of dicts containing role/content pairs (e.g., system, user, assistant).
        temperature (float, optional): Sampling temperature. Defaults to 0.2.
        output_format (Pydantic model, optional): If provided, parse the output into this format.

    Returns:
        str or parsed object: The GPT-4o output as plain text or parsed JSON (if output_format is provided).
    """
    if output_format is None:
        completion = client.beta.chat.completions.create(
            model="gpt-4o-2024-08-06",
            temperature=temperature,
            messages=messages,
        )
        return completion.choices[0].message.content
    else:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            temperature=temperature,
            response_format=output_format,
            messages=messages,
        )
        return json.loads(completion.choices[0].message.content)


def parse_atomic_statements(captions):
    """
    Call GPT to convert given captions into atomic statements.
    """
    system_message = (
        "You are an assistant that extracts fully atomic objective facts from image captions designed to output JSON. "
        "Your task is to: "
        "1. Break each caption into fully atomic statements, each expressing exactly one simple and objective fact.\n"
        "2. Each atomic statement must describe only one idea: either object existence, attribute (like color or material), position, or relationship.\n"
        "3. Do not use compound or complex sentences. Avoid words like 'and', 'but', or commas that connect multiple facts.\n"
        "4. Remove any subjective, inferred, or emotional content. Keep only visually verifiable facts.\n"
        "5. Return each atomic statement as a single plain sentence, one per line, without numbering or bullet points.\n\n"
    )

    user_message = (
        "Please convert the following captions into atomic statements.\n"
        "Caption: " + captions
    )

    return call_gpt4o(system_message, user_message, AtomicSentences)


def remove_duplicate_atomic_statements(captions):
    """
    Call GPT to remove duplicate atomic statements using system/user message separation.
    """
    system_message = (
        "You are a helpful assistant that removes semantically redundant or overlapping atomic statements. Designed to output clean and non-redundant visual facts in JSON format.\n\n"
        "Each atomic statement expresses a single visual fact from an image.\n"
        "Instructions:\n"
        "1. Only remove a statement if its **entire meaning** is fully captured by another statement.\n"
        "2. If multiple statements refer to the same object using different terms (e.g., 'bottle', 'container', 'pack'), treat them as referring to the same object and keep only the most specific and informative ones.\n"
        # "2. If two statements are phrased differently but express the **same visual fact**, keep only the clearest or most specific version.\n"
        "3. If two statements are similar in wording but describe **different facts**, keep both.\n"
        "4. Do not rewrite, rephrase, or merge statements. Just delete exact or semantically overlapping ones.\n\n"
        "Output:\n"
        "Return the final list as plain text — one sentence per line, without numbering or bullet points."
    )

    user_message = "Atomic Statements:\n" + "\n".join(captions)

    return call_gpt4o(system_message, user_message, AtomicSentences)

def remove_duplicate_atomic_statements_fewshot(captions):
    """
    Call GPT to remove duplicate atomic statements using system/user message separation with updated few-shot examples.
    """
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant that removes semantically redundant or overlapping atomic statements. "
            "Designed to output clean and non-redundant visual facts.\n\n"
            "Each atomic statement expresses a single visual fact from an image.\n"
            "Instructions:\n"
            "1. Only remove a statement if its **entire meaning** is fully captured by another statement.\n"
            "2. If multiple statements refer to the same object using different terms (e.g., 'bottle', 'container', 'pack'), treat them as referring to the same object and keep only the most specific and informative ones.\n"
            "3. If two statements are phrased differently but express the **same visual fact**, keep only the clearest or most specific version.\n"
            "4. If two statements are similar in wording but describe **different facts**, keep both.\n"
            "5. Do not rewrite, rephrase, or merge statements. Just delete exact or semantically overlapping ones.\n\n"
            "Output:\n"
            "Return the final list as plain text — one sentence per line, without numbering or bullet points."
        )
    }

    few_shot_examples = [
        # Example 1: VizWiz_train_00000283.jpg
        {
            "role": "user",
            "content": (
                "Atomic Statements:\n"
                "The canned food has a yellow label.\n"
                "The canned food has a green label.\n"
                "The canned food has a red label.\n"
                "The canned food is on a wooden surface.\n"
                "The can contains spaghetti sauce.\n"
                "The spaghetti sauce is traditional style.\n"
                "The can is 15 ounces.\n"
                "The can is 425 grams.\n"
                "There is a small can.\n"
                "The can has a picture on it.\n"
                "The picture shows red pasta sauce.\n"
                "The picture shows noodles.\n"
                "The can is on a table.\n"
                "The can is yellow.\n"
                "The can is of national brand.\n"
                "The can contains tomato paste.\n"
                "The can is sitting on a counter.\n"
                "The table is made of wood."
            )
        },
        {
            "role": "assistant",
            "content": (
                "The canned food has a yellow label.\n"
                "The canned food has a green label.\n"
                "The canned food has a red label.\n"
                "The canned food is on a wooden surface.\n"
                "The picture shows red pasta sauce.\n"
                "The spaghetti sauce is traditional style.\n"
                "The can is 15 ounces.\n"
                "The can is 425 grams.\n"
                "The can has a picture on it.\n"
                "The picture shows noodles.\n"
                "The can is of national brand."
            )
        },
        # Example 2: VizWiz_train_00000196.jpg
        {
            "role": "user",
            "content": (
                "Atomic Statements:\n"
                "There is a red box.\n"
                "The box is labeled Kaffe.\n"
                "The box is next to a washing machine.\n"
                "The package contains ground coffee.\n"
                "The coffee is raspberry flavored.\n"
                "The brand of the coffee is Gevalia Kaffe.\n"
                "The box contains Gevalia coffee.\n"
                "The coffee is Raspberry Danish flavored.\n"
                "The box is on a table."
            )
        },
        {
            "role": "assistant",
            "content": (
                "There is a red box.\n"
                "The box is next to a washing machine.\n"
                "The package contains ground coffee.\n"
                "The coffee is raspberry flavored.\n"
                "The brand of the coffee is Gevalia Kaffe.\n"
                "The box is on a table."
            )
        },
        # Example 3: VizWiz_train_00000457.jpg
        {
            "role": "user",
            "content": (
                "Atomic Statements:\n"
                "There is a bottle.\n"
                "The bottle contains shoe polish.\n"
                "The shoe polish is for leather shoes.\n"
                "The shoe polish is made by a company called Kiwi.\n"
                "There is a container.\n"
                "The container is of Kiwi leather shiner.\n"
                "The container is being held up.\n"
                "There is a product called KIWI brand shoe polish.\n"
                "The shoe polish is for leather.\n"
                "The shoe polish is labeled as 'premiere shine'.\n"
                "The shoe polish is labeled as 'ultra brilliant'.\n"
                "There is a liquid.\n"
                "The liquid can be used for food."
            )
        },
        {
            "role": "assistant",
            "content": (
                "There is a bottle.\n"
                "The bottle contains shoe polish.\n"
                "The shoe polish is for leather shoes.\n"
                "The shoe polish is made by a company called Kiwi.\n"
                "The container is being held up.\n"
                "The shoe polish is labeled as 'premiere shine'.\n"
                "The shoe polish is labeled as 'ultra brilliant'.\n"
                "There is a liquid.\n"
                "The liquid can be used for food."
            )
        }
    ]

    user_message = {
        "role": "user",
        "content": "Atomic Statements:\n" + "\n".join(captions)
    }

    messages = [system_message] + few_shot_examples + [user_message]
    return call_gpt4o_assist(messages, AtomicSentences)

def calculate_recall_gpt(T_atomics, g_atomics):
    """
    Call GPT to evaluate semantic recall between human-written (T_atomics)
    and model-generated (g_atomics) atomic statements.
    """

    system_message = (
        "You are an assistant tasked with determining the semantic equivalence between two sets of atomic sentences. "
        "The first set consists of atomic statements extracted from human-written sentences. "
        "The second set consists of atomic statements extracted from AI-generated sentences. "
        "The goal of this task is to calculate recall metrics. "
        "Definitions:\n"
        "- True Positive (TP): A human-written atomic statement whose meaning is clearly captured by at least one generated atomic statement.\n"
        "- False Negative (FN): A human-written atomic statement that is not captured or reflected in any generated statement.\n\n"
        "Instructions:\n"
        "1. For each human-written atomic statement, check whether any of the model-generated statements express the same core meaning.\n"
        "2. If the meaning is directly stated or clearly implied (without requiring external knowledge or creative inference), include the human-written statement in the True Positives (TPs).\n"
        "3. If the meaning is not directly stated or clearly implied, include the human-written statement in the False Negatives (FNs).\n"
        "4. The sum of the number of TPs and FNs should equal the number of human-written atomic statements.\n"
        "5. Use common-sense understanding when deciding if the meaning is implied — for example, if a title or visual element is described, it's reasonable to assume the cover is visible.\n"
        "6. Do NOT include any model-generated statements in the output.\n"
        "7. Avoid using outside knowledge or making assumptions beyond what is explicitly or clearly implied in the statements.\n\n"
        "Provide your response in JSON format."
    )

    user_message = (
        "Human-written atomic statements:\n"
        + "\n".join(T_atomics)
        + "\n\nGenerated atomic statements:\n"
        + "\n".join(g_atomics)
        + "\n\nReturn a JSON object in the following format:\n"
        "{\n"
        '  "TPs": [list of human-written statements that are matched],\n'
        '  "FNs": [list of human-written statements that are not matched],\n'
        '  "Match": [\n'
        '    {"T_atomic": "<human-written statement>", "g_atomic": "<matched generated statement>"},\n'
        '    ...\n'
        '  ],\n'
        '  "Counts": {"TP": number, "FN": number}\n'
        "}\n\n"
        "Again, ONLY include the human-written statements in TPs and FNs. Do NOT include any generated statements directly in those lists. "
        "Use the 'Match' field to show which human-written statements matched which generated ones. "
        "Every sentence in the `TPs` list must exactly match one of the `T_atomic` values in the `Match` field."
        "Only return the JSON object. Do NOT include any explanations or markdown formatting."
    )

    return call_gpt4o(system_message, user_message, Recall)

def calculate_precision_gpt(human_captions, g_atomics):
    """
    Call GPT to evaluate the semantic precision between human-written captions and model-generated atomic statements.
    """

    system_message = (
        "You are an assistant tasked with determining the semantic equivalence between two sets of sentences. "
        "The first set consists of human-written sentences. "
        "The second set consists of atomic statements extracted from AI-generated sentences. "
        "The goal of this task is to calculate precision metrics. "
        "Definitions:\n"
        "- True Positive (TP): A generated atomic statement that is semantically supported by, or reasonably implied by, at least one human-written caption. Exact wording is not required.\n"
        "- False Positive (FP): A generated atomic statement that introduces information not present in, or contradictory to, any of the human-written captions.\n\n"
        "Instructions:\n"
        "1. Evaluate each generated atomic statement independently.\n"
        "2. If the core meaning of a generated statement is explicitly stated or reasonably implied by any human-written caption, mark it as a True Positive (TP).\n"
        "3. If the statement includes details that are not found or are contradicted by the captions, mark it as a False Positive (FP).\n"
        "4. Accept paraphrased or partially matching statements as TP if the core meaning aligns.\n"
        "5. Do not make assumptions based on common knowledge, visual conventions, or brand familiarity unless explicitly mentioned in the captions.\n"
        "6. When listing TPs and FPs, you must use the exact original string of the generated atomic statements. Do not paraphrase, shorten, fix grammar, or modify in any way. The response must copy the sentence exactly as shown.\n\n"
        "Provide your response in JSON format."
    )

    user_message = (
        "Human-written captions:\n"
        + "\n".join(f"- {caption}" for caption in human_captions)
        + "\n\nGenerated atomic statements:\n"
        + "\n".join(f"- {statement}" for statement in g_atomics)
        + "\n\n Return a JSON object in the following format:\n"
        '- "TPs": a list of true positive generated atomic statements. Each item must be copied exactly from the list above.\n'
        '- "FPs": a list of false positive generated atomic statements. Each item must be copied exactly from the list above.\n'
        '- "Match": [\n'
        '    {"g_atomic": "<exact generated atomic statement>", "T_org": "<matching human-written caption>"},\n'
        '    ...\n'
        '  ],\n' 
        '- "Counts": a dictionary with the number of TP and FP statements.\n\n'
        "Only return the JSON object. Do NOT include any explanations or markdown formatting."
        "Use the 'Match' field to show the most relevant generated atomic caption that justifies each TP.\n"
        "Every sentence in the `TPs` list must exactly match one of the `g_atomic` values in the `Match` field."
    )

    return call_gpt4o(system_message, user_message, Precision)

def calculate_recall_gpt_fewshot(T_atomics, g_atomics):
    """
    Call GPT to evaluate semantic recall between human-written (T_atomics)
    and model-generated (g_atomics) atomic statements.
    """
    system_message = {
        "role": "system",
        "content": (
        "You are an assistant tasked with determining the semantic equivalence between two sets of atomic sentences. "
        "The first set consists of atomic statements extracted from human-written sentences. "
        "The second set consists of atomic statements extracted from AI-generated sentences. "
        "The goal of this task is to calculate recall metrics. "
        "Definitions:\n"
        "- True Positive (TP): A human-written atomic statement whose meaning is clearly captured by at least one generated atomic statement.\n"
        "- False Negative (FN): A human-written atomic statement that is not captured or reflected in any generated statement.\n\n"
        "Instructions:\n"
        "1. For each human-written atomic statement, check whether any of the model-generated statements express the same core meaning.\n"
        "2. If the meaning is directly stated or clearly implied (without requiring external knowledge or creative inference), include the human-written statement in the True Positives (TPs).\n"
        "3. If the meaning is not directly stated or clearly implied, include the human-written statement in the False Negatives (FNs).\n"
        "4. The sum of the number of TPs and FNs should equal the number of human-written atomic statements.\n"
        "5. Use common-sense understanding when deciding if the meaning is implied — for example, if a title or visual element is described, it's reasonable to assume the cover is visible.\n"
        "6. Do NOT include any model-generated statements in the output.\n"
        "7. Avoid using outside knowledge or making assumptions beyond what is explicitly or clearly implied in the statements.\n\n"
        "Provide your response in JSON format."
        )
    }

    user_message = {
        "role": "user",
        "content": (
        "Human-written atomic statements:\n"
        + "\n".join(T_atomics)
        + "\n\nGenerated atomic statements:\n"
        + "\n".join(g_atomics)
        + "\n\nReturn a JSON object in the following format:\n"
        "{\n"
        '  "TPs": [list of human-written statements that are matched],\n'
        '  "FNs": [list of human-written statements that are not matched],\n'
        '  "Match": [\n'
        '    {"T_atomic": "<human-written statement>", "g_atomic": "<matched generated statement>"},\n'
        '    ...\n'
        '  ],\n'
        '  "Counts": {"TP": number, "FN": number}\n'
        "}\n\n"
        "Again, ONLY include the human-written statements in TPs and FNs. Do NOT include any generated statements directly in those lists. "
        "Use the 'Match' field to show which human-written statements matched which generated ones. "
        "Every sentence in the `TPs` list must exactly match one of the `T_atomic` values in the `Match` field."
        "Only return the JSON object. Do NOT include any explanations or markdown formatting."
    )}


    few_shot_examples = [
        # Example 1: VizWiz_train_00000283.jpg
        {
            "role": "user",
            "content": (
                "Human-written atomic statements:\n"
                "The canned food has a yellow label.\n"
                "The canned food has a green label.\n"
                "The canned food has a red label.\n"
                "The canned food is on a wooden surface.\n"
                "The can is 15 ounces.\n"
                "The can is 425 grams.\n"
                "The can contains spaghetti sauce.\n"
                "The spaghetti sauce is traditional style.\n"
                "The can is small.\n"
                "The picture shows red pasta sauce.\n"
                "The picture shows noodles.\n"
                "The can is of national tomato paste.\n"
                "The table is made of wood."
            )
        },
        {
            "role": "assistant",
            "content": (
                '{\n'
                '  "TPs": [\n'
                '    "The canned food is on a wooden surface.",\n'
                '    "The can is 15 ounces.",\n'
                '    "The can is 425 grams.",\n'
                '    "The canned food has a yellow label.",\n'
                '    "The canned food has a red label.",\n'
                '    "The can is of national tomato paste."\n'
                '  ],\n'
                '  "FNs": [\n'
                '    "The canned food has a green label.",\n'
                '    "The can contains spaghetti sauce.",\n'
                '    "The spaghetti sauce is traditional style.",\n'
                '    "The can is small.",\n'
                '    "The picture shows red pasta sauce.",\n'
                '    "The picture shows noodles.",\n'
                '    "The table is made of wood."\n'
                '  ],\n'
                '  "Match": [\n'
                '    {"T_atomic": "The canned food is on a wooden surface.", "g_atomic": "The can is on a wooden surface."},\n'
                '    {"T_atomic": "The can is 15 ounces.", "g_atomic": "The text reads \'WT. 15 OZ. (425g)\'."},\n'
                '    {"T_atomic": "The can is 425 grams.", "g_atomic": "The text reads \'WT. 15 OZ. (425g\'."},\n'
                '    {"T_atomic": "The canned food has a yellow label.", "g_atomic": "The label has a yellow color."},\n'
                '    {"T_atomic": "The canned food has a red label.", "g_atomic": "The label has a red color."},\n'
                '    {"T_atomic": "The can is of national tomato paste.", "g_atomic": "The text includes part of the word \'TIONAL\'."}\n'
                '  ],\n'
                '  "Counts": {"TP": 6, "FN": 7}\n'
                '}'
            )
        },

        # Example 2: VizWiz_train_00000196.jpg
        {
            "role": "user",
            "content": (
                "Human-written atomic statements:\n"
                "The box is red.\n"
                "The box is placed next to a washing machine.\n"
                "There is a washing machine.\n"
                "The package is of Gevalia Kaffe brand.\n"
                "The package contains ground coffee.\n"
                "The coffee is raspberry flavored.\n"
                "The flavor is Raspberry Danish.\n"
                "The box is on a table.\n\n"
                "Generated atomic statements:\n"
                "- There is a red package.\n"
                "- The package is labeled \"GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee\".\n"
                "- There is a kitchen appliance.\n"
                "- The kitchen appliance is black and white.\n"
                "- The kitchen appliance is partially visible."
            )
        },
        {
            "role": "assistant",
            "content": (
                '{\n'
                '  "TPs": [\n'
                '    "The box is red.",\n'
                '    "There is a washing machine.",\n'
                '    "The package is of Gevalia Kaffe brand.",\n'
                '    "The coffee is raspberry flavored.",\n'
                '    "The flavor is Raspberry Danish."\n'
                '  ],\n'
                '  "FNs": [\n'
                '    "The box is placed next to a washing machine.",\n'
                '    "The package contains ground coffee.",\n'
                '    "The box is on a table."\n'
                '  ],\n'
                '  "Match": [\n'
                '    {"T_atomic": "The box is red.", "g_atomic": "There is a red package."},\n'
                '    {"T_atomic": "There is a washing machine.", "g_atomic": "There is a kitchen appliance."},\n'
                '    {"T_atomic": "The package is of Gevalia Kaffe brand.", "g_atomic": "The package is labeled \"GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee\"."},\n'
                '    {"T_atomic": "The coffee is raspberry flavored.", "g_atomic": "The package is labeled \"GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee\"."},\n'
                '    {"T_atomic": "The flavor is Raspberry Danish.", "g_atomic": "The package is labeled \"GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee\"."}\n'
                '  ],\n'
                '  "Counts": {"TP": 5, "FN": 3}\n'
                '}'
            )
        }, 
        # Example 3: VizWiz_train_00000457.jpg
        {
            "role": "user",
            "content": (
                "Human-written atomic statements:\n"
                "There is a bottle.\n"
                "The bottle contains shoe polish.\n"
                "The shoe polish is for leather shoes.\n"
                "The shoe polish is made by a company called Kiwi.\n"
                "The container is being held up.\n"
                "The shoe polish is labeled as cuir.\n"
                "The shoe polish is labeled as premiere shine.\n"
                "The shoe polish is labeled as ultra brilliant.\n"
                "There is a liquid.\n"
                "The liquid is used for food.\n\n"
                "Generated atomic statements:\n"
                "- A person is holding a container.\n"
                "- The container is a Kiwi brand product.\n"
                "- The product is for leather care.\n"
                "- The product is designed for shoe care.\n"
                "- The container is labeled \"Premiere Shine Ultra-Brillant.\"\n"
                "- The container is primarily red.\n"
                "- The container is primarily black.\n"
                "- There is a sponge applicator at the top of the container."
            )
        },
        {
            "role": "assistant",
            "content": (
                '{\n'
                '  "TPs": [\n'
                '    "The container is being held up.",\n'
                '    "The shoe polish is labeled as premiere shine.",\n'
                '    "The shoe polish is labeled as ultra brilliant."\n'
                '  ],\n'
                '  "FNs": [\n'
                '    "There is a bottle.",\n'
                '    "The bottle contains shoe polish.",\n'
                '    "The shoe polish is for leather shoes.",\n'
                '    "The shoe polish is made by a company called Kiwi.",\n'
                '    "The shoe polish is labeled as cuir.",\n'
                '    "There is a liquid.",\n'
                '    "The liquid is used for food."\n'
                '  ],\n'
                '  "Match": [\n'
                '    {"T_atomic": "The container is being held up.", "g_atomic": "A person is holding a container."},\n'
                '    {"T_atomic": "The shoe polish is labeled as premiere shine.", "g_atomic": "The container is labeled \"Premiere Shine Ultra-Brillant.\""},\n'
                '    {"T_atomic": "The shoe polish is labeled as ultra brilliant.", "g_atomic": "The container is labeled \"Premiere Shine Ultra-Brillant.\""}\n'
                '  ],\n'
                '  "Counts": {"TP": 3, "FN": 7}\n'
                '}'
            )
        }

    ]

    messages = [system_message] + few_shot_examples + [user_message]
    return call_gpt4o_assist(messages, Recall)

def calculate_precision_gpt_fewshot(human_captions, g_atomics):
    system_message = {
        "role": "system",
        "content": (
            "You are an assistant tasked with determining the semantic equivalence between two sets of sentences. "
            "The first set consists of human-written sentences. "
            "The second set consists of atomic statements extracted from AI-generated sentences. "
            "The goal of this task is to calculate precision metrics. "
            "Definitions:\n"
            "- True Positive (TP): A generated atomic statement that is semantically supported by, or reasonably implied by, at least one human-written caption. Exact wording is not required.\n"
            "- False Positive (FP): A generated atomic statement that introduces information not present in, or contradictory to, any of the human-written captions.\n\n"
            "Instructions:\n"
            "1. Evaluate each generated atomic statement independently.\n"
            "2. If the core meaning of a generated statement is explicitly stated or reasonably implied by any human-written caption, mark it as a True Positive (TP).\n"
            "3. If the statement includes details that are not found or are contradicted by the captions, mark it as a False Positive (FP).\n"
            "4. Accept paraphrased or partially matching statements as TP if the core meaning aligns.\n"
            "5. Do not make assumptions based on common knowledge, visual conventions, or brand familiarity unless explicitly mentioned in the captions.\n"
            "6. Avoid inferring visual details such as color or design purely from product names or brand recognition.\n"
            "7. When listing TPs and FPs, you must use the exact original string of the generated atomic statements. Do not paraphrase, shorten, fix grammar, or modify in any way. The response must copy the sentence exactly as shown.\n\n"
            "Provide your response in JSON format."
        )
    }

    user_message = {
        "role": "user",
        "content": (
            "Human-written captions:\n"
            + "\n".join(f"- {caption}" for caption in human_captions)
            + "\n\nGenerated atomic statements:\n"
            + "\n".join(f"- {statement}" for statement in g_atomics)
            + "\n\nReturn a JSON object in the following format:\n"
            '{\n'
            '  "TPs": [list of true positive generated atomic statements],\n'
            '  "FPs": [list of false positive generated atomic statements],\n'
            '  "Match": [\n'
            '    {"g_atomic": "<exact generated atomic statement>", "T_org": "<matching human-written caption>"},\n'
            '    ...\n'
            '  ],\n'
            '  "Counts": {"TP": number, "FP": number}\n'
            '}\n\n'
            "Only return the JSON object. Do NOT include any explanations or markdown formatting.\n"
            "Use the 'Match' field to show the most relevant human-written caption that justifies each TP.\n"
            "Every sentence in the `TPs` list must exactly match one of the `g_atomic` values in the `Match` field."
        )
    }

    few_shot_examples = [
        # Example 1: VizWiz_train_00000283.jpg
        {
            "role": "user",
            "content": (
                "Human-written captions:\n"
                "- A canned food that has a yellow, green, and red label on a wooden surface.\n"
                "- A 15 ounce (425g) can of traditional style spaghetti sauce.\n"
                "- A small can with a picture of red pasta sauce on top of noodles on a table.\n"
                "- A photo of a yellow can of national tomato paste sitting on a counter.\n"
                "- A can with a yellow and red label sits on a wooden table.\n\n"
                "Generated atomic statements:\n"
                "- There is a can.\n"
                "- The can is on a wooden surface.\n"
                "- The can has visible text.\n"
                "- The text reads 'WT. 15 OZ. (425g)'.\n"
                "- The text includes part of the word 'TIONAL'.\n"
                "- The label has a yellow color.\n"
                "- The label has a red color."
            )
        },
        {
            "role": "assistant",
            "content": (
                '{\n'
                '  "TPs": [\n'
                '    "There is a can.",\n'
                '    "The can is on a wooden surface.",\n'
                '    "The text reads \'WT. 15 OZ. (425g)\'",\n'
                '    "The label has a yellow color.",\n'
                '    "The label has a red color."\n'
                '  ],\n'
                '  "FPs": [\n'
                '    "The text includes part of the word \'TIONAL\'.",\n'
                '    "The can has visible text."\n'
                '  ],\n'
                '  "Match": [\n'
                '    {"g_atomic": "There is a can.", "T_org": "A can with a yellow and red label sits on a wooden table."},\n'
                '    {"g_atomic": "The can is on a wooden surface.", "T_org": "A canned food that has a yellow, green, and red label on a wooden surface."},\n'
                '    {"g_atomic": "The text reads \'WT. 15 OZ. (425g)\'", "T_org": "A 15 ounce (425g) can of traditional style spaghetti sauce."},\n'
                # '    {"g_atomic": "The text includes part of the word \'TIONAL\'.", "T_org": "A photo of a yellow can of national tomato paste sitting on a counter."},\n'
                '    {"g_atomic": "The label has a yellow color.", "T_org": "A can with a yellow and red label sits on a wooden table."},\n'
                '    {"g_atomic": "The label has a red color.", "T_org": "A can with a yellow and red label sits on a wooden table."}\n'
                '  ],\n'
                '  "Counts": {"TP": 6, "FP": 1}\n'
                '}'
            )
        },


        # Example 2: VizWiz_train_00000196.jpg
        {
            "role": "user",
            "content": (
                "Human-written captions:\n"
                "- A red box of Kaffe placed next to a washing machine.\n"
                "- A package of Gevalia Kaffe brand raspberry flavored ground coffee\n"
                "- A box of Raspberry Danish flavored Gevalia coffee.\n"
                "- Red box with raspberry flavored ground coffee on table.\n\n"
                "Generated atomic statements:\n"
                "- There is a red package.\n"
                "- The package is labeled \"GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee\".\n"
                "- There is a kitchen appliance.\n"
                "- The kitchen appliance is black and white.\n"
                "- The kitchen appliance is partially visible."
            )
        },
        {
            "role": "assistant",
            "content": (
                '{\n'
                '  "TPs": [\n'
                '    "There is a red package.",\n'
                '    "The package is labeled \\"GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee\\".",\n'
                '    "There is a kitchen appliance.",\n'
                '  ],\n'
                '  "FPs": [\n'
                '    "The kitchen appliance is black and white.",\n'
                '    "The kitchen appliance is partially visible."\n'
                '  ],\n'
                '  "Match": [\n'
                '    {"g_atomic": "There is a red package.", "T_org": "Red box with raspberry flavored ground coffee on table."},\n'
                '    {"g_atomic": "The package is labeled \\"GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee\\".", "T_org": "A package of Gevalia Kaffe brand raspberry flavored ground coffee"},\n'
                '    {"g_atomic": "There is a kitchen appliance.", "T_org": "A red box of Kaffe placed next to a washing machine."},\n'
                # '    {"g_atomic": "The kitchen appliance is black and white.", "T_org": "A red box of Kaffe placed next to a washing machine."},\n'
                # '    {"g_atomic": "The kitchen appliance is partially visible.", "T_org": "A red box of Kaffe placed next to a washing machine."}\n'
                '  ],\n'
                '  "Counts": {"TP": 5, "FP": 0}\n'
                '}'
            )
        },

        # Example 3: VizWiz_train_00000457.jpg
        {
            "role": "user",
            "content": (
                "Human-written captions:\n"
                "- I believe this is a bottle of leather shoe polish for premium shine made by a company called kiwi.\n"
                "- A container of Kiwi leather shiner is being held up.\n"
                "- KIWI brand shoe polish - leather cuir premiere shine ultra brilliant\n"
                "- some type of liquid stuff you can use for food\n\n"
                "Generated atomic statements:\n"
                "- A person is holding a container.\n"
                "- The container is a Kiwi brand product.\n"
                "- The product is for leather care.\n"
                "- The product is designed for shoe care.\n"
                "- The container is labeled \"Premiere Shine Ultra-Brillant.\"\n"
                "- The container is primarily red.\n"
                "- The container is primarily black.\n"
                "- There is a sponge applicator at the top of the container."
            )
        },
        {
            "role": "assistant",
            "content": (
                '{\n'
                '  "TPs": [\n'
                '    "A person is holding a container.",\n'
                '    "The container is a Kiwi brand product.",\n'
                '    "The product is for leather care.",\n'
                '    "The product is designed for shoe care.",\n'
                '    "The container is labeled \\"Premiere Shine Ultra-Brillant.\\""\n'
                '  ],\n'
                '  "FPs": [\n'
                '    "The container is primarily red.",\n'
                '    "The container is primarily black.",\n'
                '    "There is a sponge applicator at the top of the container."\n'
                '  ],\n'
                '  "Match": [\n'
                '    {"g_atomic": "A person is holding a container.", "T_org": "A container of Kiwi leather shiner is being held up."},\n'
                '    {"g_atomic": "The container is a Kiwi brand product.", "T_org": "KIWI brand shoe polish - leather cuir premiere shine ultra brilliant"},\n'
                '    {"g_atomic": "The product is for leather care.", "T_org": "I believe this is a bottle of leather shoe polish for premium shine made by a company called kiwi."},\n'
                '    {"g_atomic": "The product is designed for shoe care.", "T_org": "I believe this is a bottle of leather shoe polish for premium shine made by a company called kiwi."},\n'
                '    {"g_atomic": "The container is labeled \\"Premiere Shine Ultra-Brillant.\\"", "T_org": "KIWI brand shoe polish - leather cuir premiere shine ultra brilliant"}\n'
                '  ],\n'
                '  "Counts": {"TP": 5, "FP": 3}\n'
                '}'
            )
        }


    ]

    messages = [system_message] + few_shot_examples + [user_message]
    return call_gpt4o_assist(messages, Precision)


# This is only for testing the prompt
def generate_atomic_statement_part1(org_caption):
    T_atomics = []

    for item in tqdm(org_caption):
        # Filter out human captions
        human_captions = [
            hc["caption"]
            for hc in item["human_captions"]
            if hc["caption"]
            != "Quality issues are too severe to recognize visual content."
        ]

        human_atomic_captions = []
        for hc in human_captions:
            human_atomic_captions.append(parse_atomic_statements(hc))

        # print(json.dumps(human_atomic_captions, indent=4, ensure_ascii=False))
        total_sentence = [
            s for output in human_atomic_captions for s in output["atomic_captions"]
        ]

        T_atomics.append(total_sentence)

    return T_atomics

# This is only for testing the prompt
def generate_atomic_statement_part_reduce(atomic_captions):
    T_atomics = [] 
    for item in tqdm(atomic_captions): 
        parsed_statements = item["evaluation"]["previous_cap_f1"]["parsed_atomics"]
        # print(json.dumps(parsed_statements, indent=4))

        human_result = remove_duplicate_atomic_statements(parsed_statements)
        T_atomics.append(human_result)

    return T_atomics

# This is only for testing the prompt
def generate_atomic_statement_reduce_fewshot(atomic_captions):
    T_atomics = [] 
    for item in tqdm(atomic_captions): 
        parsed_statements = item["evaluation"]["previous_cap_f1"]["parsed_atomics"]
        # print(json.dumps(parsed_statements, indent=4))

        human_result = remove_duplicate_atomic_statements_fewshot(parsed_statements)
        T_atomics.append(human_result)

    return T_atomics

# This is only for testing the prompt
def generate_atomic_statement_partG(atomic_captions):
    g_atomics = []

    for item in tqdm(atomic_captions):
        # Model captions
        model_results = []
        for mc in item["model_captions"]:
            model_name = mc["model_name"]
            model_caption = mc["caption"]
            atomic_result = parse_atomic_statements(model_caption)

            model_results.append(
                {
                    "model_name": model_name,
                    "atomic_captions": atomic_result["atomic_captions"],
                }
            )

        g_atomics.append(model_results)

    return g_atomics

def generate_atomic_statement_getG(atomic_captions):
    g_atomics_list = []

    for item in tqdm(atomic_captions): 
        model_to_atomics = item["evaluation"]["previous_cap_f1"].get("g_atomics", {})
        
        parsed_statements = []
        for model_name, atomics in model_to_atomics.items():
            parsed_statements.append({
                "model_name": model_name,
                "atomic_captions": atomics  # 바로 리스트로 넣어줌
            })

        g_atomics_list.append(parsed_statements)

    return g_atomics_list



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
    parsed_T = []

    for item in tqdm(org_caption[:limit]):
        # Filter out human captions
        human_captions = [
            hc["caption"]
            for hc in item["human_captions"]
            if hc["caption"]
            != "Quality issues are too severe to recognize visual content."
        ]

        human_atomic_captions = []
        for hc in human_captions:
            human_atomic_captions.append(parse_atomic_statements(hc))

        # print(json.dumps(human_atomic_captions, indent=4, ensure_ascii=False))
        total_sentence = [
            s for output in human_atomic_captions for s in output["atomic_captions"]
        ]
        human_result = remove_duplicate_atomic_statements_fewshot(total_sentence)
        parsed_T.append(total_sentence)

        # Model captions
        model_results = []
        for mc in item["model_captions"]:
            model_name = mc["model_name"]
            model_caption = mc["caption"]
            atomic_result = parse_atomic_statements(model_caption)

            model_results.append(
                {
                    "model_name": model_name, 
                    "atomic_captions": atomic_result["atomic_captions"],
                }
            )

        T_atomics.append(human_result)
        g_atomics.append(model_results)

    return T_atomics, g_atomics, parsed_T

def evaluate_single_instance(model_name, T_atomics, T_original, g_captions, print_mode=False):
    if print_mode:
        print("T atomics \n", json.dumps(T_atomics, indent=4, ensure_ascii=False))
        print("T original \n", json.dumps(T_original, indent=4, ensure_ascii=False))
        print(f"{model_name} g atomics \n", json.dumps(g_captions, indent=4, ensure_ascii=False))

    recall_result = calculate_recall_gpt_fewshot(T_atomics, g_captions)
    precision_result = calculate_precision_gpt_fewshot(T_original, g_captions)

    check_consistency(
        model_name=model_name,
        T_atomics=T_atomics,
        g_captions=g_captions,
        recall_TP=recall_result["TPs"],
        recall_FN=recall_result["FNs"],
        precision_TP=precision_result["TPs"],
        precision_FP=precision_result["FPs"]
    )

    return {
        "model_name": model_name,
        "recall": recall_result,
        "precision": precision_result,
    }

def evaluate_matching_file(parsed_dataset, print_mode=False):
    eval_outputs = []

    for item in tqdm(parsed_dataset):
        T_org = item["human_captions"]
        T_atomics = item["evaluation"]["cap_f1"]["T_atomics"]
        g_atomics = item["evaluation"]["cap_f1"]["g_atomics"]

        model_outputs = []
        for model_name, g_captions in g_atomics.items():
            output = evaluate_single_instance(
                model_name, T_atomics, T_org, g_captions, print_mode
            )
            model_outputs.append(output)

        eval_outputs.append(model_outputs)

    return eval_outputs


def evaluate_matching(T_org, T_atomics, g_atomics, print_mode=False):
    eval_outputs = []

    for i in tqdm(range(len(T_atomics))):
        T_atomic = T_atomics[i]["atomic_captions"]
        g_atomic = g_atomics[i]
        T_original = T_org[i]

        model_outputs = []
        for g_item in g_atomic:
            model_name = g_item["model_name"]
            g_captions = g_item["atomic_captions"]

            output = evaluate_single_instance(
                model_name, T_atomic, T_original, g_captions, print_mode
            )
            model_outputs.append(output)

        eval_outputs.append(model_outputs)

    return eval_outputs

def calculate_cap_f1(evaluation):
    total_output = []
    for item in tqdm(evaluation):  # item is a list of model evaluations
        model_outputs = []
        for model_scores in item:
            model_name = model_scores["model_name"]
            if (
                "Counts" not in model_scores["precision"]
                or model_scores["precision"]["Counts"] is None
            ):
                continue

            precision_TP = model_scores["precision"]["Counts"]["TP"]
            precision_FP = model_scores["precision"]["Counts"]["FP"]
            recall_TP = model_scores["recall"]["Counts"]["TP"]
            recall_FN = model_scores["recall"]["Counts"]["FN"]

            precision = (
                precision_TP / (precision_TP + precision_FP)
                if (precision_TP + precision_FP) != 0
                else 0
            )
            recall = (
                recall_TP / (recall_TP + recall_FN)
                if (recall_TP + recall_FN) != 0
                else 0
            )
            cap_f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) != 0
                else 0
            )

            # print(f"precision: {precision:.4f}, recall: {recall:.4f}, cap_f1: {cap_f1:.4f}")

            model_outputs.append(
                {
                    "model_name": model_name,
                    "recall": recall,
                    "precision": precision,
                    "cap_f1": cap_f1,
                }
            )

        total_output.append(model_outputs)
    return total_output


# def calculate_cap_f1_file(evaluation):
#     for item in evaluation:
#         for model_name, model_scores in item["evaluation"]["cap_f1"]["scores"].items():
#             print(f"Model: {model_name}")
#             print(json.dumps(model_scores, indent=4, ensure_ascii=False))

#             precision_TP = model_scores["precision"]["Counts"]["TP"]
#             precision_FP = model_scores["precision"]["Counts"]["FP"]
#             recall_TP = model_scores["recall"]["Counts"]["TP"]
#             recall_FN = model_scores["recall"]["Counts"]["FN"]

#             precision = precision_TP / (precision_TP + precision_FP) if (precision_TP + precision_FP) != 0 else 0
#             recall = recall_TP / (recall_TP + recall_FN) if (recall_TP + recall_FN) != 0 else 0
#             cap_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

#             print(f"precision: {precision:.4f}, recall: {recall:.4f}, cap_f1: {cap_f1:.4f}")


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
            cider_score, _ = cider_metric.compute_score(
                cider_references, cider_predictions
            )

            eval_outputs.append(
                {
                    "model_name": model_name,
                    "BLEU": bleu.compute(
                        predictions=predictions, references=references
                    ),
                    "METEOR": meteor.compute(
                        predictions=predictions, references=references
                    ),
                    "ROUGE": rouge.compute(
                        predictions=predictions, references=references
                    ),
                    "CIDEr": cider_score,
                }
            )

    return eval_outputs


def main():
    print("see pipeline.ipynb")


if __name__ == "__main__":
    main()
