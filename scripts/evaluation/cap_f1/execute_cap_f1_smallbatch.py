from datetime import datetime
import os
import argparse

from cap_f1 import LLMClient, AtomicProcessor, ResultsRepo
from fewshot_examples import (
    FEWSHOT_DEDUP_MESSAGES,
    FEWSHOT_RECALL_MESSAGES,
    FEWSHOT_PRECISION_MESSAGES,
)

# 1) build API + processor (inject few-shot examples if you want)
llm = LLMClient()
proc = AtomicProcessor(
    llm,
    fewshot_dedup=FEWSHOT_DEDUP_MESSAGES,  # or None
    fewshot_recall=FEWSHOT_RECALL_MESSAGES,  # or None
    fewshot_precision=FEWSHOT_PRECISION_MESSAGES,  # or None
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, help="Input file", required=True)
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to output intermediatary and final results.",
        default="./results",
    )
    parser.add_argument("--limit", type=int, default=4, help="batch size limit")
    return parser.parse_args()


def main():
    # parse the arguments
    args = parse_args()

    # setup output folder and file metadata
    # this is where intermediate and final results will be saved
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M")

    # create folder to save the results
    folder_path = f"{args.output_path}/{timestamp}"
    os.makedirs(folder_path, exist_ok=True)

    # 2) load data
    print("Loading caption dataset...")
    dataset = ResultsRepo.read_json(args.input_file)

    # 3) generate atomics
    print("Generating atomic statements using gpt-4o...")
    T_atomics, g_atomics, parsed_T = proc.generate_atomic_statement(
        dataset, limit=args.limit
    )

    # 3.1) save intermediate
    print("Saving intermediate results...")
    all_human_captions = []
    for item in dataset[: args.limit]:
        # Filter out human captions that are mention quality issues
        human_captions = [
            hc["caption"]
            for hc in item["human_captions"]
            if hc["caption"]
            != "Quality issues are too severe to recognize visual content."
        ]
        all_human_captions.append(human_captions)

    ResultsRepo.save_results_json(
        output_path=f"{folder_path}/intermediate_{timestamp}.json",
        org_dataset=dataset,
        parsed_T=parsed_T,
        T_atomics=T_atomics,
        g_atomics=g_atomics,
        T_org=all_human_captions,
        limit=args.limit,
    )

    # 4) evaluate and get recall and precision
    # - match human caption to model caption
    # - create recall and precision data
    print("Evaluating atomic statements...")

    eval_out = proc.evaluate_matching(
        T_org=all_human_captions, T_atomics=T_atomics, g_atomics=g_atomics
    )

    # 4.1) save evaluation results
    ResultsRepo.save_results_json(
        output_path=f"{folder_path}/eval_{timestamp}.json",
        update_existing=f"{folder_path}/intermediate_{timestamp}.json",
        metadata=eval_out,
        limit=args.limit,
    )

    # 5) calculate cap f1 score
    print("calculating F1 scores...")
    cap_scores = proc.calculate_cap_f1(eval_out)

    # 5.1) save cap f1 score results
    print("Saving scored results...")
    ResultsRepo.save_results_json(
        output_path=f"{folder_path}/final_{timestamp}.json",
        update_existing=f"{folder_path}/eval_{timestamp}.json",
        evaluations=cap_scores,
        limit=args.limit,
    )

    # 6) Final JSON → CSV
    print("Saving final results into csv...")
    ResultsRepo.export_final_csv(
        json_path=f"{folder_path}/final_{timestamp}.json",
        csv_path=f"{folder_path}/final_{timestamp}.csv",
        # model_keys={"gpt":"gpt-4o-2024-08-06", "molmo":"Molmo-7B-O-0924", "llama":"Llama-3.2-11B-Vision-Instruct"}
    )


if __name__ == "__main__":
    main()
