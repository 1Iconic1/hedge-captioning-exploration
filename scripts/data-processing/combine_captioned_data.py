"""
Combine the captioning data from the different models into a single file.

Example usage:
python combine_captioned_data.py \
    --llama ../../data/study-2-output/labeled-data/llama-caption-output/Llama-3.2-11B-Vision-Instruct_caption-output_7304-images_start-0_end-7304_2025-03-29_00:43:55.json \
    --molmo ../../data/study-2-output/labeled-data/molmo-caption-output/Molmo-7B-O-0924_caption-output_7304-images_start-0_end-7304_2025-03-28_22:45:36.json \
    --gpt ../../data/study-2-output/labeled-data/gpt4o-caption-output/gpt-4o-2024-08-06_caption-output_7304-images_start-0_end-7304_2025-03-29_13:29:15.json \
    --output ../../data/study-2-output/labeled-data/combined-caption-output/

python combine_captioned_data.py \
    --llama ../../data/study-2-output/labeled-data/high-quality-images/llama-caption-output/Llama-3.2-11B-Vision-Instruct_caption-output_5428-images_start-0_end-5428_2025-04-08_18:31_fixed.json \
    --molmo ../../data/study-2-output/labeled-data/high-quality-images/molmo-caption-output/Molmo-7B-O-0924_caption-output_5428-images_start-0_end-5428_2025-04-08_17:42_fixed.json \
    --gpt ../../data/study-2-output/labeled-data/high-quality-images/gpt4o-caption-output/gpt-4o-2024-08-06_caption-output_5428-images_start-0_end-5428_2025-04-09_00:15_fixed.json \
    --output ../../data/study-2-output/labeled-data/combined-caption-output/
"""

import argparse
import os
import copy
import json
from datetime import datetime


def load_captioned_data(file_path):
    """Load the captioning data from the given file path.

    Args:
        file_path (str): The path to the file containing the captioning data. File will be a JSON.

    Returns:
        list of dict: The captioning data.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def combine_captioned_data(gpt_data, llama_data, molmo_data):
    """Combine the captioning data from the different models into a single file.

    Args:
        gpt_data (list of dict): The GPT-captioned data.
        llama_data (list of dict): The Llama-captioned data.
        molmo_data (list of dict): The Molmo-captioned data.

    Returns:
        list of dict: The combined data.
    """
    # sort all data by image_id
    gpt_data = sorted(gpt_data, key=lambda x: x["image_id"])
    llama_data = sorted(llama_data, key=lambda x: x["image_id"])
    molmo_data = sorted(molmo_data, key=lambda x: x["image_id"])

    # combine the data
    output = copy.deepcopy(gpt_data)
    for i in range(len(llama_data)):
        # verify that the image_id is the same
        if output[i]["image_id"] == llama_data[i]["image_id"]:
            output[i]["model_captions"].append(llama_data[i]["model_captions"][0])
    for i in range(len(molmo_data)):
        # verify that the image_id is the same
        if output[i]["image_id"] == molmo_data[i]["image_id"]:
            output[i]["model_captions"].append(molmo_data[i]["model_captions"][0])

    # remove things we don't need
    for item in output:
        if "id" in item:
            del item["id"]
    return output


def save_combined_data(data, output_path):
    """Save the combined data to the given output path.

    Args:
        data (list of dict): The combined data.
        output_path (str): The path to save the combined data.

    Returns:
        str: The path to the saved file.
    """
    # check that the output path exists
    os.makedirs(output_path, exist_ok=True)

    # save the data
    output_file_path = os.path.join(
        output_path,
        f"combined-caption-output_{len(data)}-images_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json",
    )
    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=4)

    return output_file_path


def main():
    # get files from user using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama", type=str, help="File for Llama-captioned data.")
    parser.add_argument("--molmo", type=str, help="File for Molmo-captioned data.")
    parser.add_argument("--gpt", type=str, help="File for GPT-captioned data.")
    parser.add_argument("--output", type=str, help="Output directory path.")
    args = parser.parse_args()

    # load the data from the files
    llama_data = load_captioned_data(args.llama)
    molmo_data = load_captioned_data(args.molmo)
    gpt_data = load_captioned_data(args.gpt)

    combined_data = combine_captioned_data(gpt_data, llama_data, molmo_data)
    output_file_path = save_combined_data(combined_data, args.output)
    print(f"Combined data saved to {output_file_path}")


if __name__ == "__main__":
    main()
