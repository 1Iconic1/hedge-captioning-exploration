"""
Runs all evaluation metrics on the captions and outputs a json file with the results.

Usage:
python evaluate_captions.py \
    --input ../../data/study-2-output/labeled-data/combined-caption-output/combined-caption-output_7304-images_2025-03-29_21:40:00.json \
    --output-dir ../../data/study-2-output/labeled-data/evaluation-results \
    --start 0 \
    --end 1
"""

import argparse
import os
import json
from datetime import datetime

import evaluate
import torch
from bert_score import score
from tqdm import tqdm

# setup pytorch for BERTScore
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for multi-GPU systems, force single GPU
if torch.cuda.is_available():
    device_type = "cuda:0"  # force single, first GPU
elif torch.backends.mps.is_available():
    device_type = "mps"
else:
    device_type = "cpu"
print(f"Using device: {device_type}")


def load_data(input_file):
    """
    Load the data from the input file

    Args:
        input_file (str): path to the input file

    Returns:
        list of dict: list of dictionaries containing captioning data
    """
    with open(input_file, "r") as f:
        data = json.load(f)
    return data


def filter_data(data):
    """
    Remove data where Text = False

    Args:
        data (list of dict): list of dictionaries containing captioning data

    Returns:
        list of dict: list of dictionaries containing captioning data
    """
    return [d for d in data if d["text_detected"]]


def evaluate_captions(data, model_type="microsoft/deberta-xlarge-mnli", lang="en"):
    """
    Evaluate the captions using all metrics. Update data by reference.

    Args:
        data (list of dict): list of dictionaries containing captioning data
        model_type (str): type of model to use for BERTScore evaluation
        lang (str): language of the captions for BERTScore evaluation

    Returns:
        None
    """
    # initialize the metrics
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    for image in tqdm(data):
        curr_references = [
            caption["caption"]
            for caption in image["human_captions"]
            if caption["caption"]
            != "Quality issues are too severe to recognize visual content."
        ]
        curr_output = {}
        for model in image["model_captions"]:
            curr_model_name = model["model_name"]
            curr_output[curr_model_name] = {}
            curr_candidate = [model["caption"]]

            # compute BERTScore
            P, R, F1 = score(
                curr_candidate,
                [curr_references],
                model_type=model_type,
                lang=lang,
                device=device_type,
            )
            curr_output[curr_model_name]["bertscore"] = {
                "scores": {
                    "precision": float(P[0]),
                    "recall": float(R[0]),
                    "f1": float(F1[0]),
                }
            }

            # compute BLEU
            for order in range(1, 5):
                bleu_score = bleu.compute(
                    predictions=curr_candidate,
                    references=[curr_references],
                    max_order=order,
                )
                curr_output[curr_model_name][f"bleu-{order}"] = bleu_score

            # compute METEOR
            meteor_score = meteor.compute(
                predictions=curr_candidate, references=[curr_references]
            )
            curr_output[curr_model_name]["meteor"] = meteor_score

            # compute ROUGE
            rouge_score = rouge.compute(
                predictions=curr_candidate, references=[curr_references]
            )
            curr_output[curr_model_name]["rouge"] = rouge_score

        # check if evaluation exists and save score
        if "evaluation" not in image:
            image["evaluation"] = {}
        image["evaluation"] = curr_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input JSON file path.")
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for JSON file."
    )
    parser.add_argument(
        "--start", type=int, help="Start index of the data to evaluate."
    )
    parser.add_argument("--end", type=int, help="End index of the data to evaluate.")
    args = parser.parse_args()

    # load the data from the files
    data = load_data(args.input)
    data = filter_data(data)
    start_idx = args.start if args.start else 0
    end_idx = args.end if args.end else len(data)
    data = data[start_idx:end_idx]

    # evaluate the captions
    print(
        f"Evaluating data from {start_idx} to {end_idx} (total images: {len(data)})..."
    )
    evaluate_captions(data)

    # check if output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # create an output path by using the file name from the input path and output directory
    output_path = os.path.join(
        args.output_dir,
        f"evaluation_results_{len(data)}-images_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.json",
    )

    # save the data to the output path
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
