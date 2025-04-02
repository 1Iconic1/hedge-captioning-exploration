"""
This script is used to generate a batch file for captioning images using the OpenAI Batch API.

Example usage:
python generate_captioning_batch_file.py \
    --start 0 \
    --end 5
"""

import sys

sys.path.append("..")

import os
import json
import pandas as pd
import argparse
from datetime import datetime

from constants import get_prompt
from data_loader import generate_target_dataset, filter_dataset

# global variables for script
model_name = "gpt-4o-2024-08-06"
prompt = get_prompt()
temperature = 1.0
top_p = 1.0
max_output_tokens = 300
store = False


def generate_batch_line(image_id, image_url):
    """
    Generate a batch line for the OpenAI Batch API.

    Args:
        image_id (int): The ID of the image to caption.
        image_url (str): The URL of the image to caption.
        model_name (str): The name of the model to use for captioning.

    Returns:
        dict: A dictionary containing the batch line for the OpenAI Batch API.

    Example:
    {
        "custom_id":"request-1",
        "method":"POST",
        "url":"/v1/chat/completions",
        "body":{
            "model":"gpt-3.5-turbo-0125",
            "messages":[
                {
                    "role":"system",
                    "content":"You are a helpful assistant."
                },
                {
                    "role":"user",
                    "content":"Hello world!"
                }
            ],
            "max_tokens":1000
        }
    }
    """
    return {
        "custom_id": f"request-{image_id}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model_name,
            "input": [
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": image_url,
                        }
                    ],
                },
            ],
            "reasoning": {},
            "tools": [],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
            "store": store,
            "text": {"format": {"type": "text"}},
        },
    }


def generate_batch_file(dataset):
    """
    Generate a batch file for captioning images using the OpenAI Batch API.

    Args:
        dataset (list of dicts): A list of dictionaries containing the dataset to caption.

    Returns:
        list: A list of strings containing the batch line for each image for the OpenAI Batch API.
    """
    batch_file = []
    for index, row in enumerate(dataset):
        batch_file.append(
            json.dumps(generate_batch_line(row["image_id"], row["vizwiz_url"]))
        )
    return batch_file


def save_batch_file(batch_list, output_path):
    """
    Save the list of batch instructions to the output path as a .jsonl file.

    Args:
        batch_list (list of str): A list of strings containing the batch line for each image for the OpenAI Batch API.
        output_path (str): The path to save the batch file.

    Returns:
        str: output_path of the saved batch file
    """
    with open(output_path, "w") as f:
        f.write("\n".join(batch_list))
    return output_path


def main():
    """
    Main function to generate a batch file for captioning images using the OpenAI Batch API.
    """
    # make sure we have a batch_input and batch_jobs directory
    os.makedirs("./batch_input", exist_ok=True)
    os.makedirs("./batch_jobs", exist_ok=True)

    # load data
    print("Generating target dataset...")
    dataset_to_caption = generate_target_dataset(
        "../../data/caption-dataset/annotations/train.json",
        "../../data/image-quality-assessment/annotations/train.json",
    )
    dataset_to_caption = filter_dataset(pd.DataFrame.from_dict(dataset_to_caption))
    print(f"Filtered dataset of {len(dataset_to_caption)} images.")

    # get the start and ending index of the dataset to caption
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="",
    )

    args = parser.parse_args()
    start_index = args.start
    end_index = args.end if args.end is not None else len(dataset_to_caption)
    output_path = (
        args.output
        if args.output != ""
        else f"./batch_input/{model_name}-captioning-batch-input_start-{start_index}_end-{end_index}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.jsonl"
    )

    # generate batch file
    print(f"Generating batch file for {start_index} to {end_index} images...")
    batch_file = generate_batch_file(dataset_to_caption[start_index:end_index])

    save_batch_file(batch_file, output_path)
    print(f"Batch file generated for {len(batch_file)} images: {output_path}")


if __name__ == "__main__":
    main()
