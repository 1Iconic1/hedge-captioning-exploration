"""
Corrects the format of the evaluation JSON files.

Original format (model --> metric --> scores):
"evaluation": {
    "gpt-4o-2024-08-06": {
        "bertscore": {
            "scores": {
                "precision": 0.6953458189964294,
                "recall": 0.800594687461853,
                "f1": 0.7392169237136841
            }
        },
        "bleu-1": {...},
        ...
    },
    "Llama-3.2-11B-Vision-Instruct": {...},
    "Molmo-7B-O-0924": {...}
}

Correct format (metric --> model --> scores):
"evaluation": {
    "bertscore": {
        "gpt-4o-2024-08-06": {
            "scores": {
                "precision": 0.6953458189964294,
                "recall": 0.800594687461853,
                "f1": 0.7392169237136841
            }
        },
        "Llama-3.2-11B-Vision-Instruct": {...},
        "Molmo-7B-O-0924": {...}
    },
    "bleu-1": {...},
    ...
}

Usage:
python fix_evaluation_json_format.py \
    --input ../../data/study-2-output/labeled-data/evaluation-results/evaluation_results_5432-images_2025-04-03_11:27.json \
    --output-path ../../data/study-2-output/labeled-data/evaluation-results
"""

import argparse
import copy
import json
import os


def fix_evaluation_json_format(data):
    """
    Fixes the format of the evaluation portion of captioned data.

    The original format is model --> metric --> scores.
    The correct format is metric --> model --> scores.

    Args:
        data (list): A list of dictionaries containing the evaluation data.

    Returns:
        list: A list of dictionaries containing the evaluation data with the correct format.
    """
    # to make sure we don't modify the original data, we'll make a deep copy
    output_data = copy.deepcopy(data)

    # iterate over each item in the data
    for item in output_data:
        # create a new evaluation dict that we'll populate with the correct format
        new_evaluation = {}

        # we currently have model --> metric --> scores
        # we want to go metric --> model --> scores
        # for each model, extract the metric
        for model in item["evaluation"].keys():
            for metric in item["evaluation"][model].keys():
                # extract the scores from model --> metric --> scores
                scores = copy.deepcopy(item["evaluation"][model][metric])

                # create a new dict for each metric
                if metric not in new_evaluation:
                    new_evaluation[metric] = {}

                # now we have metric --> model --> scores
                new_evaluation[metric][model] = scores

        # update the item with the new evaluation format
        # it's probably not necessary to make a deep copy here, but we'll do it to make sure we don't have issues with data references
        item["evaluation"] = copy.deepcopy(new_evaluation)

    return output_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    # load data
    with open(args.input, "r") as f:
        data = json.load(f)

    # fix evaluation JSON format
    fixed_data = fix_evaluation_json_format(data)

    # save fixed data
    filename = os.path.basename(args.input)
    remove_extension = os.path.splitext(filename)[0]
    output_path = os.path.join(args.output_path, f"{remove_extension}_fixed.json")
    with open(output_path, "w") as f:
        json.dump(fixed_data, f, indent=4)

    print(f"Fixed data saved to {output_path}")


if __name__ == "__main__":
    main()
