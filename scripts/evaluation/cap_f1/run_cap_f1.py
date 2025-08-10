from datetime import datetime
import os

from scripts.evaluation.cap_f1.cap_f1 import LLMClient, AtomicProcessor, ResultsRepo
from scripts.evaluation.cap_f1.fewshot_examples import (
    FEWSHOT_DEDUP_MESSAGES,
    FEWSHOT_RECALL_MESSAGES,
    FEWSHOT_PRECISION_MESSAGES,
)

# 0) Set environment variables
LIMIT = 1
# for filename
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M")

#create folder to save the results
folder_path = f"results/{timestamp}"
os.makedirs(folder_path, exist_ok=True)

# 1) build API + processor (inject few-shot examples if you want)
llm = LLMClient()
proc = AtomicProcessor(
    llm,
    fewshot_dedup=FEWSHOT_DEDUP_MESSAGES,           # or None
    fewshot_recall=FEWSHOT_RECALL_MESSAGES,         # or None
    fewshot_precision=FEWSHOT_PRECISION_MESSAGES,   # or None
)

# 2) load data
print("Loading caption dataset...")
dataset = ResultsRepo.read_json("one_data.json")

# 3) generate atomics
print("Generating atomic statements using gpt-4o...")
T_atomics, g_atomics, parsed_T = proc.generate_atomic_statement(dataset, limit=LIMIT)

# 4) (optional) save intermediate
print("Saving intermediate results...")
ResultsRepo.save_results_json(
    output_path=f"{folder_path}/intermediate_{timestamp}.json",
    org_dataset=dataset,
    parsed_T=parsed_T,
    T_atomics=T_atomics,
    g_atomics=g_atomics,
    T_org=[[hc["caption"] for hc in it["human_captions"]] for it in dataset[:LIMIT]],
)

# 5) evaluate + F1
print("Evaluating atomic statements...")
eval_out = proc.evaluate_matching(
    T_org=[[hc["caption"] for hc in it["human_captions"]] for it in dataset[:LIMIT]],
    T_atomics=T_atomics,
    g_atomics=g_atomics
)

proc.save_results_json(
    output_path=f"{folder_path}/eval_{timestamp}.json",
    update_existing=f"{folder_path}/intermediate_{timestamp}.json",
    metadata=eval_out, 
    limit=LIMIT
)

print("calculating F1 scores...")
cap_scores = proc.calculate_cap_f1(eval_out)

# 6) save scored results
print("Saving scored results...")
ResultsRepo.save_results_json(
    output_path=f"{folder_path}/final_{timestamp}.json",
    update_existing=f"{folder_path}/eval_{timestamp}.json",
    evaluations=cap_scores,
)

# 7) JSON → CSV
print("Saving final results into csv...")
ResultsRepo.export_final_csv(
    json_path=f"{folder_path}/final_{timestamp}.json",
    csv_path=f"{folder_path}/final_{timestamp}.csv",
    # model_keys={"gpt":"gpt-4o-2024-08-06", "molmo":"Molmo-7B-O-0924", "llama":"Llama-3.2-11B-Vision-Instruct"}
)