"""
Corrects the format of the human caption in JSON files when not using data loader.

Usage:
Molmo
python fix_human_caption.py \
    --input ../../data/study-2-output/labeled-data/high-quality-images/molmo-caption-output/Molmo-7B-O-0924_caption-output_5428-images_start-0_end-5428_2025-04-08_17:42.json \
    --output-path ../../data/study-2-output/labeled-data/high-quality-images/molmo-caption-output/

Llama
python fix_human_caption.py \
    --input ../../data/study-2-output/labeled-data/high-quality-images/llama-caption-output/Llama-3.2-11B-Vision-Instruct_caption-output_5428-images_start-0_end-5428_2025-04-08_18:31.json \
    --output-path ../../data/study-2-output/labeled-data/high-quality-images/llama-caption-output/

GPT4o
python fix_human_caption.py \
    --input ../../data/study-2-output/labeled-data/high-quality-images/gpt4o-caption-output/gpt-4o-2024-08-06_caption-output_5428-images_start-0_end-5428_2025-04-09_00:15.json \
    --output-path ../../data/study-2-output/labeled-data/high-quality-images/gpt4o-caption-output/
"""

import argparse
import copy
import json
import os


def fix_human_caption(data):
    """
    Fixes the format of the human caption in JSON files.

    The original format is a list of dictionaries with the following keys:
    - "caption" (str): the caption
    - "is_precanned" (bool): whether the caption is precanned
    - "is_rejected" (bool): whether the caption is rejected

    The correct format is a list of dictionaries with the following keys:
    - "human_captions" (list of dict): list of dictionaries containing human captions
    - "model_captions" (list of dict): list of dictionaries containing model captions

    Args:
        data (list): A list of dictionaries containing the evaluation data.

    Returns:
        list: A list of dictionaries containing the evaluation data with the correct format.
    """
    output = copy.deepcopy(data)
    # expand captions, is_precanned, and is_rejected into individual columns
    for index, row in enumerate(output):
        curr_captions = row["caption"]
        curr_precanned = row["is_precanned"]
        curr_rejected = row["is_rejected"]

        # expand captions
        human_captions = []
        for caption_index in range(0, len(curr_captions)):
            curr_human_caption = {
                "caption": curr_captions[caption_index],
                "is_precanned": curr_precanned[caption_index],
                "is_rejected": curr_rejected[caption_index],
            }
            human_captions.append(curr_human_caption)

        output[index]["human_captions"] = human_captions

        # remove old rows
        del output[index]["caption"]
        del output[index]["is_precanned"]
        del output[index]["is_rejected"]

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    # load data
    with open(args.input, "r") as f:
        data = json.load(f)

    # fix human caption
    fixed_data = fix_human_caption(data)

    # save fixed data
    filename = os.path.basename(args.input)
    remove_extension = os.path.splitext(filename)[0]
    output_path = os.path.join(args.output_path, f"{remove_extension}_fixed.json")
    with open(output_path, "w") as f:
        json.dump(fixed_data, f, indent=4)

    print(f"Fixed data saved to {output_path}")


if __name__ == "__main__":
    main()
