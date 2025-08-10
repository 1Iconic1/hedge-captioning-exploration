# refactor_pipeline.py
from __future__ import annotations
import json, os, time, random
from typing import List, Dict, Any, Optional, Type, Union
from collections import Counter
from pydantic import BaseModel
from tqdm import tqdm
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

# ========
# Schemas
# ========
class AtomicSentences(BaseModel):
    atomic_captions: List[str]

class RecallCounts(BaseModel):
    TP: int; FN: int

class RecallMatchPair(BaseModel):
    T_atomic: str; g_atomic: str

class Recall(BaseModel):
    TPs: List[str]; FNs: List[str]
    Match: List[RecallMatchPair]; Counts: RecallCounts

class PrecisionMatchPair(BaseModel):
    g_atomic: str; T_org: str

class PrecisionCounts(BaseModel):
    TP: int; FP: int

class Precision(BaseModel):
    TPs: List[str]; FPs: List[str]
    Match: List[PrecisionMatchPair]; Counts: PrecisionCounts

# =========
# Constants
# =========
OPENAI_MODEL = "gpt-4o-2024-08-06"
SENTINEL_BAD_QUALITY = "Quality issues are too severe to recognize visual content."
METRIC_CAP = "cap_f1"

# =========
# IO layer
# =========
class ResultsRepo:
    @staticmethod
    def read_json(path: str, keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
            Read JSON file and extract only the specified keys.

            Inputs:
            - path: path to JSON file
            - keys: list of keys to extract from each item in the JSON

            Output:
            - list(dictionary)
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not keys:
            return data
        return [{key: item[key] for key in keys if key in item} for item in data]

    @staticmethod
    def save_results_json(
        output_path: str,
        org_dataset: Optional[List[Dict[str, Any]]] = None,
        parsed_T: Optional[List[List[str]]] = None,
        T_atomics: Optional[List[Dict[str, List[str]]]] = None,
        g_atomics: Optional[List[List[Dict[str, Any]]]] = None,
        T_org: Optional[List[List[str]]] = None,
        metadata: Optional[List[List[Dict[str, Any]]]] = None,
        meta_start: int = 0,
        evaluations: Optional[List[List[Dict[str, Any]]]] = None,
        metric_name: str = METRIC_CAP,
        update_existing: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> None:
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
            for i in range(min(len(results) - meta_start, len(metadata))):
                idx = i + meta_start
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
                results[idx]["evaluation"].setdefault(metric_name, {})["metadata"] = metric_scores

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

    @staticmethod
    def _format_matches(match_list):
        lines = []
        for m in match_list or []:
            if isinstance(m, dict):
                if "T_atomic" in m and "g_atomic" in m:
                    lines.append(f'{m["T_atomic"]} : {m["g_atomic"]}')
                elif "g_atomic" in m and "T_org" in m:
                    lines.append(f'{m["g_atomic"]} : {m["T_org"]}')
                else:
                    lines.append(json.dumps(m, ensure_ascii=False))
            else:
                lines.append(str(m))
        return "\n".join(lines)
    
    @staticmethod
    def export_final_csv(
        json_path: str,
        csv_path: str,
        *,
        viz_base_url: str = "https://vizwiz.cs.colorado.edu/VizWiz_visualization_img/",
        model_keys: dict = None,
    ) -> None:
        import csv

        if model_keys is None:
            model_keys = {
                "gpt": "gpt-4o-2024-08-06",
                "molmo": "Molmo-7B-O-0924",
                "llama": "Llama-3.2-11B-Vision-Instruct",
            }

        fieldnames = [
            "image",
            "link",
            "T_org",
            "parsed_T",
            "T_atomics",
            "gpt_caption",
            "gpt_g_atomics",
            "gpt_recall_TPs",
            "gpt_recall_Matches",
            "gpt_recall_FNs",
            "gpt_precision_TPs",
            "gpt_precision_Matches",
            "gpt_precision_FPs",
            "molmo_caption",
            "molmo_g_atomics",
            "molmo_recall_TPs",
            "molmo_recall_Matches",
            "molmo_recall_FNs",
            "molmo_precision_TPs",
            "molmo_precision_Matches",
            "molmo_precision_FPs",
            "llama_caption",
            "llama_g_atomics",
            "llama_recall_TPs",
            "llama_recall_Matches",
            "llama_recall_FNs",
            "llama_precision_TPs",
            "llama_precision_Matches",
            "llama_precision_FPs",
            "gpt_recall",
            "gpt_precision",
            "gpt_capf1",
            "molmo_recall",
            "molmo_precision",
            "molmo_capf1",
            "llama_recall",
            "llama_precision",
            "llama_capf1",
        ]

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for item in data:
                file_name = item.get("file_name", "")
                cap_f1 = item.get("evaluation", {}).get("cap_f1", {})
                scores = cap_f1.get("scores", {})
                metadata = cap_f1.get("metadata", {})
                t_atomics = cap_f1.get("T_atomics", [])
                parsed_T = cap_f1.get("parsed_atomics", [])
                T_org = cap_f1.get("T_org", [])

                # model_name -> caption 매핑(원본 코드는 인덱스 사용: 더 안전하게 이름으로 매핑)
                captions_by_model = {}
                for mc in item.get("model_captions", []):
                    captions_by_model[mc.get("model_name", "")] = mc.get("caption", "")

                row = {
                    "image": file_name,
                    "link": f'=HYPERLINK("{viz_base_url}{file_name}", "{file_name}")' if file_name else "",
                    "T_org": "\n".join(T_org),
                    "parsed_T": "\n".join(parsed_T),
                    "T_atomics": "\n".join(t_atomics),
                    "gpt_caption": captions_by_model.get(model_keys["gpt"], ""),
                    "molmo_caption": captions_by_model.get(model_keys["molmo"], ""),
                    "llama_caption": captions_by_model.get(model_keys["llama"], ""),
                    "gpt_g_atomics": "", "molmo_g_atomics": "", "llama_g_atomics": "",
                    "gpt_recall_TPs": "", "molmo_recall_TPs": "", "llama_recall_TPs": "",
                    "gpt_recall_Matches": "", "molmo_recall_Matches": "", "llama_recall_Matches": "",
                    "gpt_recall_FNs": "", "molmo_recall_FNs": "", "llama_recall_FNs": "",
                    "gpt_precision_TPs": "", "molmo_precision_TPs": "", "llama_precision_TPs": "",
                    "gpt_precision_Matches": "", "molmo_precision_Matches": "", "llama_precision_Matches": "",
                    "gpt_precision_FPs": "", "molmo_precision_FPs": "", "llama_precision_FPs": "",
                    "gpt_recall": scores.get(model_keys["gpt"], {}).get("recall"),
                    "gpt_precision": scores.get(model_keys["gpt"], {}).get("precision"),
                    "gpt_capf1": scores.get(model_keys["gpt"], {}).get("cap_f1"),
                    "molmo_recall": scores.get(model_keys["molmo"], {}).get("recall"),
                    "molmo_precision": scores.get(model_keys["molmo"], {}).get("precision"),
                    "molmo_capf1": scores.get(model_keys["molmo"], {}).get("cap_f1"),
                    "llama_recall": scores.get(model_keys["llama"], {}).get("recall"),
                    "llama_precision": scores.get(model_keys["llama"], {}).get("precision"),
                    "llama_capf1": scores.get(model_keys["llama"], {}).get("cap_f1"),
                }

                for short_name, model_key in model_keys.items():
                    g_atomics_list = cap_f1.get("g_atomics", {}).get(model_key, [])
                    row[f"{short_name}_g_atomics"] = "\n".join(g_atomics_list)

                    recall = metadata.get(model_key, {}).get("recall", {})
                    row[f"{short_name}_recall_TPs"] = "\n".join(recall.get("TPs", []))
                    row[f"{short_name}_recall_FNs"] = "\n".join(recall.get("FNs", []))
                    row[f"{short_name}_recall_Matches"] = ResultsRepo._format_matches(recall.get("Match", []))

                    precision = metadata.get(model_key, {}).get("precision", {})
                    row[f"{short_name}_precision_TPs"] = "\n".join(precision.get("TPs", []))
                    row[f"{short_name}_precision_FPs"] = "\n".join(precision.get("FPs", []))
                    row[f"{short_name}_precision_Matches"] = ResultsRepo._format_matches(precision.get("Match", []))

                writer.writerow(row)

        print(f"CSV file saved to: {csv_path}")

    @staticmethod
    def merge_json_chunks(output_file, file_pattern):
        import glob
        merged_data = []

        for filename in sorted(glob.glob(file_pattern)):
            with open(filename, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        merged_data.extend(data)
                    elif isinstance(data, dict):
                        merged_data.append(data)
                except Exception as e:
                    print(f"Failed to read {filename}: {e}")

        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(merged_data, out_f, indent=2, ensure_ascii=False)

        print(f"Merged {len(merged_data)} entries into {output_file}")


# =================
# OpenAI/API layer
# =================
class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = OPENAI_MODEL, temperature: float = 0.2):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature

    # ---- retry helper ----
    def _retry(self, fn, retries=3, base=0.8, jitter=0.2):
        last = None
        for i in range(retries):
            try:
                return fn()
            except (RateLimitError, APITimeoutError, APIError) as e:
                last = e
                if i == retries - 1:
                    raise
                time.sleep(base * (2 ** i) + random.random() * jitter)
        if last:
            raise last

    # ---- low-level callers ----
    def _chat(self, messages: List[Dict[str, str]]) -> str:
        def _do():
            return self.client.beta.chat.completions.create(
                model=self.model, temperature=self.temperature, messages=messages
            )
        completion = self._retry(_do)
        return completion.choices[0].message.content

    def _chat_parse(self, messages: List[Dict[str, str]], response_model: Type[BaseModel]) -> Dict[str, Any]:
        def _do():
            return self.client.beta.chat.completions.parse(
                model=self.model, temperature=self.temperature, response_format=response_model, messages=messages
            )
        completion = self._retry(_do)
        parsed = getattr(completion.choices[0].message, "parsed", None)
        if parsed is not None:
            # Some SDKs give a Pydantic instance; convert to plain dict
            return json.loads(parsed.model_dump_json()) if hasattr(parsed, "model_dump_json") else parsed
        return json.loads(completion.choices[0].message.content)

    # ---- prompts ----
    def parse_atomic_statements(self, caption: str) -> Dict[str, List[str]]:
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
            "Caption: " + caption
        )
        msgs = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return self._chat_parse(msgs, AtomicSentences)

    def dedup_atomics(self, captions: List[str], fewshot_examples: Optional[List[Dict[str, str]]] = None) -> Dict[str, List[str]]:

        if fewshot_examples:
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
                ),
            }
            user_message = {
                "role": "user",
                "content": "Atomic Statements:\n" + "\n".join(captions),
            }
            msgs = [system_message] + fewshot_examples + [user_message]
            return self._chat_parse(msgs, AtomicSentences)

        # No fewshot examples, use default system message
        system_message = (
            "You are a helpful assistant that removes semantically redundant or overlapping atomic statements. Designed to output clean and non-redundant visual facts in JSON format.\n\n"
            "Each atomic statement expresses a single visual fact from an image.\n"
            "Instructions:\n"
            "1. Only remove a statement if its **entire meaning** is fully captured by another statement.\n"
            "2. If multiple statements refer to the same object using different terms (e.g., 'bottle', 'container', 'pack'), treat them as referring to the same object and keep only the most specific and informative ones.\n"
            "3. If two statements are similar in wording but describe **different facts**, keep both.\n"
            "4. Do not rewrite, rephrase, or merge statements. Just delete exact or semantically overlapping ones.\n\n"
            "Output:\n"
            "Return the final list as plain text — one sentence per line, without numbering or bullet points."
        )
        user_message = "Atomic Statements:\n" + "\n".join(captions)
        msgs = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return self._chat_parse(msgs, AtomicSentences)

    def recall_json(self, T_atomics: List[str], g_atomics: List[str], fewshot_examples: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:

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
            ),
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
                "    ...\n"
                "  ],\n"
                '  "Counts": {"TP": number, "FN": number}\n'
                "}\n\n"
                "Again, ONLY include the human-written statements in TPs and FNs. Do NOT include any generated statements directly in those lists. "
                "Use the 'Match' field to show which human-written statements matched which generated ones. "
                "Every sentence in the `TPs` list must exactly match one of the `T_atomic` values in the `Match` field."
                "Only return the JSON object. Do NOT include any explanations or markdown formatting."
            ),
        }

        msgs = [system_message] + (fewshot_examples or []) + [user_message]
        return self._chat_parse(msgs, Recall)

    def precision_json(self, human_captions: List[str], g_atomics: List[str], fewshot_examples: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
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
            ),
        }

        user_message = {
            "role": "user",
            "content": (
                "Human-written captions:\n"
                + "\n".join(f"- {caption}" for caption in human_captions)
                + "\n\nGenerated atomic statements:\n"
                + "\n".join(f"- {statement}" for statement in g_atomics)
                + "\n\nReturn a JSON object in the following format:\n"
                "{\n"
                '  "TPs": [list of true positive generated atomic statements],\n'
                '  "FPs": [list of false positive generated atomic statements],\n'
                '  "Match": [\n'
                '    {"g_atomic": "<exact generated atomic statement>", "T_org": "<matching human-written caption>"},\n'
                "    ...\n"
                "  ],\n"
                '  "Counts": {"TP": number, "FP": number}\n'
                "}\n\n"
                "Only return the JSON object. Do NOT include any explanations or markdown formatting.\n"
                "Use the 'Match' field to show the most relevant human-written caption that justifies each TP.\n"
                "Every sentence in the `TPs` list must exactly match one of the `g_atomic` values in the `Match` field."
            ),
        }

        msgs = [system_message] + (fewshot_examples or []) + [user_message]
        return self._chat_parse(msgs, Precision)

# =================
# Processing layer
# =================
class AtomicProcessor:
    def __init__(self, llm: LLMClient, fewshot_dedup: Optional[List[Dict[str, str]]] = None,
                 fewshot_recall: Optional[List[Dict[str, str]]] = None,
                 fewshot_precision: Optional[List[Dict[str, str]]] = None):
        self.llm = llm
        self.fewshot_dedup = fewshot_dedup
        self.fewshot_recall = fewshot_recall
        self.fewshot_precision = fewshot_precision

    def generate_atomic_statement(self, org_caption: List[Dict[str, Any]], limit: int = 2):
        """
        Returns:
          T_atomics: list[dict] -> {"atomic_captions":[...]} after dedup
          g_atomics: list[list[{"model_name":..., "atomic_captions":[...]}]]
          parsed_T: list[list[str]] -> raw, pre-dedup flattened atomics
        """
        T_atomics, g_atomics, parsed_T = [], [], []
        for item in tqdm(org_caption[:limit]):
            # --- humans ---
            human_caps = [hc["caption"] for hc in item["human_captions"] if hc["caption"] != SENTINEL_BAD_QUALITY]
            human_atomic_flat: List[str] = []
            for cap in human_caps:
                out = self.llm.parse_atomic_statements(cap)  # {"atomic_captions":[...]}
                human_atomic_flat.extend(out["atomic_captions"])
            parsed_T.append(human_atomic_flat)

            dedup = self.llm.dedup_atomics(human_atomic_flat, fewshot_examples=self.fewshot_dedup)
            T_atomics.append(dedup)

            # --- models ---
            model_results = []
            for mc in item["model_captions"]:
                mn, text = mc["model_name"], mc["caption"]
                gout = self.llm.parse_atomic_statements(text)
                model_results.append({"model_name": mn, "atomic_captions": gout["atomic_captions"]})
            g_atomics.append(model_results)

        return T_atomics, g_atomics, parsed_T

    def evaluate_single_instance(self, model_name: str, T_atomics: List[str], T_original: List[str],
                                 g_captions: List[str], print_mode: bool = False) -> Dict[str, Any]:
        if print_mode:
            print("T atomics\n", json.dumps(T_atomics, indent=2, ensure_ascii=False))
            print("T original\n", json.dumps(T_original, indent=2, ensure_ascii=False))
            print(f"{model_name} g atomics\n", json.dumps(g_captions, indent=2, ensure_ascii=False))

        recall = self.llm.recall_json(T_atomics, g_captions, fewshot_examples=self.fewshot_recall)
        precision = self.llm.precision_json(T_original, g_captions, fewshot_examples=self.fewshot_precision)

        self._check_consistency(model_name, T_atomics, g_captions,
                                recall_TP=recall["TPs"], recall_FN=recall["FNs"],
                                precision_TP=precision["TPs"], precision_FP=precision["FPs"])
        return {"model_name": model_name, "recall": recall, "precision": precision}

    def evaluate_matching(self, T_org: List[List[str]], T_atomics: List[Dict[str, List[str]]],
                          g_atomics: List[List[Dict[str, Any]]], print_mode: bool = False) -> List[List[Dict[str, Any]]]:
        outputs: List[List[Dict[str, Any]]] = []
        for i in tqdm(range(len(T_atomics))):
            human_atomic = T_atomics[i]["atomic_captions"]
            human_original = T_org[i]
            model_list = g_atomics[i]
            per_models = []
            for g_item in model_list:
                name = g_item["model_name"]
                g_caps = g_item["atomic_captions"]
                per_models.append(self.evaluate_single_instance(name, human_atomic, human_original, g_caps, print_mode))
            outputs.append(per_models)
        return outputs

    @staticmethod
    def calculate_cap_f1(evaluation: List[List[Dict[str, Any]]]) -> List[List[Dict[str, float]]]:
        total_output = []
        for item in tqdm(evaluation):
            per_models = []
            for m in item:
                # guard missing counts
                prec_counts = m["precision"].get("Counts") or {}
                rec_counts = m["recall"].get("Counts") or {}
                if not prec_counts or not rec_counts:
                    continue
                pTP, pFP = prec_counts.get("TP", 0), prec_counts.get("FP", 0)
                rTP, rFN = rec_counts.get("TP", 0), rec_counts.get("FN", 0)
                precision = pTP / (pTP + pFP) if (pTP + pFP) else 0.0
                recall = rTP / (rTP + rFN) if (rTP + rFN) else 0.0
                cap_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
                per_models.append({"model_name": m["model_name"], "recall": recall, "precision": precision, "cap_f1": cap_f1})
            total_output.append(per_models)
        return total_output

    @staticmethod
    def _check_consistency(model_name: str, T_atomics: List[str], g_captions: List[str],
                           recall_TP: List[str], recall_FN: List[str],
                           precision_TP: List[str], precision_FP: List[str]) -> None:
        # multiset-aware
        if Counter(recall_TP) + Counter(recall_FN) != Counter(T_atomics):
            print(f"[{model_name}] Recall mismatch: len T={len(T_atomics)} vs TP+FN={len(recall_TP)+len(recall_FN)}")
        if Counter(precision_TP) + Counter(precision_FP) != Counter(g_captions):
            print(f"[{model_name}] Precision mismatch: len G={len(g_captions)} vs TP+FP={len(precision_TP)+len(precision_FP)}")
