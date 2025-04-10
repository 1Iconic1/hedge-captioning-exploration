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
import glob
import json
import csv
import math

from datetime import datetime
from multiprocessing import Pool

# load cap_f1
from cap_f1 import *


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
    T_atomics, g_atomics, parsed_T = generate_atomic_statement(subset, limit=LIMIT)
    save_results_json(
        output_path=f"{folder_path}/parsed_caption_{timestamp}_chunk{chunk_id}.json",
        org_dataset=subset,
        T_atomics=T_atomics,
        g_atomics=g_atomics,
        parsed_T=parsed_T,
        T_org=human_subset,
        limit=LIMIT,
    )

    # Step 2: Match human & generated
    metadata = evaluate_matching(human_subset, T_atomics, g_atomics)
    save_results_json(
        output_path=f"{folder_path}/recall_precision_{timestamp}_chunk{chunk_id}.json",
        update_existing=f"{folder_path}/parsed_caption_{timestamp}_chunk{chunk_id}.json",
        metadata=metadata,
        limit=LIMIT,
    )

    # Step 3: Cap F1
    evaluation = calculate_cap_f1(metadata)
    save_results_json(
        output_path=f"{folder_path}/final_{timestamp}_chunk{chunk_id}.json",
        update_existing=f"{folder_path}/recall_precision_{timestamp}_chunk{chunk_id}.json",
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


def merge_json_chunks(output_file, file_pattern):
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


def format_as_csv(folder_path, timestamp):
    def format_matches(match_list):
        lines = []
        for m in match_list:
            if "T_atomic" in m and "g_atomic" in m:
                lines.append(f'{m["T_atomic"]} : {m["g_atomic"]}')
            elif "g_atomic" in m and "T_org" in m:
                lines.append(f'{m["g_atomic"]} : {m["T_org"]}')
            else:
                lines.append(str(m))  # fallback for unexpected format
        return "\n".join(lines)

    json_path = f"{folder_path}/__final_{timestamp}_merged.json"
    csv_path = f"{folder_path}/__final_{timestamp}_merged.csv"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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

            model_keys = {
                "gpt": "gpt-4o-2024-08-06",
                "molmo": "Molmo-7B-O-0924",
                "llama": "Llama-3.2-11B-Vision-Instruct",
            }

            row = {
                "image": file_name,
                "link": f'=HYPERLINK("https://vizwiz.cs.colorado.edu/VizWiz_visualization_img/{file_name}", "{file_name}")',
                "T_org": "\n".join(T_org),
                "parsed_T": "\n".join(parsed_T),
                "T_atomics": "\n".join(t_atomics),
                "gpt_caption": item["model_captions"][0]["caption"],
                "molmo_caption": item["model_captions"][2]["caption"],
                "llama_caption": item["model_captions"][1]["caption"],
                "gpt_g_atomics": "",
                "molmo_g_atomics": "",
                "llama_g_atomics": "",
                "gpt_recall_TPs": "",
                "molmo_recall_TPs": "",
                "llama_recall_TPs": "",
                "gpt_recall_Matches": "",
                "molmo_recall_Matches": "",
                "llama_recall_Matches": "",
                "gpt_recall_FNs": "",
                "molmo_recall_FNs": "",
                "llama_recall_FNs": "",
                "gpt_precision_TPs": "",
                "molmo_precision_TPs": "",
                "llama_precision_TPs": "",
                "gpt_precision_Matches": "",
                "molmo_precision_Matches": "",
                "llama_precision_Matches": "",
                "gpt_precision_FPs": "",
                "molmo_precision_FPs": "",
                "llama_precision_FPs": "",
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
                row[f"{short_name}_recall_Matches"] = format_matches(
                    recall.get("Match", [])
                )

                precision = metadata.get(model_key, {}).get("precision", {})
                row[f"{short_name}_precision_TPs"] = "\n".join(precision.get("TPs", []))
                row[f"{short_name}_precision_FPs"] = "\n".join(precision.get("FPs", []))
                row[f"{short_name}_precision_Matches"] = format_matches(
                    precision.get("Match", [])
                )

            writer.writerow(row)

    print(f"CSV file saved to: {csv_path}")


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
    input_data = read_json(args.input_file)

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
    merge_json_chunks(
        output_file=f"{folder_path}/__final_{timestamp}_merged.json",
        file_pattern=f"{folder_path}/final_{timestamp}_chunk*.json",
    )

    # format the merged json as csv and save
    format_as_csv(folder_path, timestamp)


if __name__ == "__main__":
    main()
