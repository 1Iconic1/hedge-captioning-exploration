"""
This script is used to execute the Cap F1 evaluation in parallell for a captioned dataset.

Usage:
python execute_cap_f1_parallel.py \
    --input-file <path-to-input-file> \
    --num-workers <number-of-workers> \
    --output-path <path-to-output-folder> \
    --start <start-index> \
    --end <end-index>

Example usage:
python execute_cap_f1_parallel.py \
    --input-file ../../data/study-2-output/final-evaluated-captions/low-quality_evaluation_5432-images_2025-04-10_15:29.json \
    --num-workers 8 \
    --output-path ./results \
    --start 0 \
    --end 8
"""

# library for cap_f1
import argparse
import os
import math

from datetime import datetime
from multiprocessing import Pool

# load cap_f1
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
    fewshot_dedup=FEWSHOT_DEDUP_MESSAGES,           # or None
    fewshot_recall=FEWSHOT_RECALL_MESSAGES,         # or None
    fewshot_precision=FEWSHOT_PRECISION_MESSAGES,   # or None
)


def process_batch(
    start_idx,
    end_idx,
    org_caption_dataset,
    all_human_captions,
    folder_path,
    timestamp,
    chunk_id,
):
    subset = org_caption_dataset[start_idx:end_idx]
    LIMIT = len(subset)
    human_subset = all_human_captions[start_idx:end_idx]

    # Step 1: Parse atomics
    T_atomics, g_atomics, parsed_T = proc.generate_atomic_statement(subset, limit=LIMIT)
    ResultsRepo.save_results_json(
        output_path=f"{folder_path}/intermediate_{timestamp}_chunk{chunk_id}.json",
        org_dataset=subset,
        T_atomics=T_atomics,
        g_atomics=g_atomics,
        parsed_T=parsed_T,
        T_org=human_subset,
        limit=LIMIT,
    )

    # Step 2: Match human & generated
    metadata = proc.evaluate_matching(human_subset, T_atomics, g_atomics)
    ResultsRepo.save_results_json(
        output_path=f"{folder_path}/eval_{timestamp}_chunk{chunk_id}.json",
        update_existing=f"{folder_path}/intermediate_{timestamp}_chunk{chunk_id}.json",
        metadata=metadata,
        limit=LIMIT,
    )

    # Step 3: get Cap F1 score
    evaluation = proc.calculate_cap_f1(metadata)
    ResultsRepo.save_results_json(
        output_path=f"{folder_path}/final_{timestamp}_chunk{chunk_id}.json",
        update_existing=f"{folder_path}/eval_{timestamp}_chunk{chunk_id}.json",
        evaluations=evaluation,
        limit=LIMIT,
    )


def run_parallel_processing(
    org_caption_dataset, all_human_captions, folder_path, timestamp, num_workers=32
):
    total = len(org_caption_dataset)
    chunk_size = math.ceil(total / num_workers)

    with Pool(processes=num_workers) as pool:
        jobs = []
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total)
            jobs.append(
                pool.apply_async(
                    process_batch,
                    (
                        start_idx,
                        end_idx,
                        org_caption_dataset,
                        all_human_captions,
                        folder_path,
                        timestamp,
                        i,
                    ),
                )
            )

        for job in jobs:
            job.get()




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, help="Input file", required=True)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to output intermediatary and final results.",
        default="./results",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, help="End index")
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

    # load the input dataset
    print("Loading caption dataset...")
    input_data = ResultsRepo.read_json(args.input_file)

    # extract human captions from the input dataset
    all_human_captions = []
    for item in input_data:
        # Filter out human captions
        human_captions = [
            hc["caption"]
            for hc in item["human_captions"]
            if hc["caption"]
            != "Quality issues are too severe to recognize visual content."
        ]
        all_human_captions.append(human_captions)

    # start executing
    start = args.start
    end = args.end if args.end else len(input_data)

    run_parallel_processing(
        input_data[start:end],
        all_human_captions[start:end],
        folder_path,
        timestamp,
        num_workers=args.num_workers,
    )

    # merge the json chunks and save output as json
    ResultsRepo.merge_json_chunks(
        output_file=f"{folder_path}/final_{timestamp}_merged.json",
        file_pattern=f"{folder_path}/final_{timestamp}_chunk*.json",
    )

    # JSON → CSV
    print("Saving final results into csv...")
    ResultsRepo.export_final_csv(
        json_path=f"{folder_path}/final_{timestamp}_merged.json",
        csv_path=f"{folder_path}/final_{timestamp}.csv",
        # model_keys={"gpt":"gpt-4o-2024-08-06", "molmo":"Molmo-7B-O-0924", "llama":"Llama-3.2-11B-Vision-Instruct"}
    )

if __name__ == "__main__":
    main()
