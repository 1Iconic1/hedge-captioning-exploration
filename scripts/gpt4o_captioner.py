"""
Usage
For high-quality images:
python gpt4o_captioner.py \
    --input-file "../data/study-2-input/high-quality-image_iq-1_text-no-text.json" \
    --output-dir "../data/study-2-output/labeled-data/high-quality-images/gpt4o-caption-output" \
    --scratch-path "../data/scratch/study-2/gpt4o-caption-output" \
    --start 0 \
    --end 12
"""

# Libraries
import argparse
import json
import math
import pandas as pd
import os
import copy
from multiprocessing import Pool
from datetime import datetime

from openai import OpenAI
from tqdm import tqdm

# custom imports
from constants import get_prompt
from data_loader import generate_target_dataset, filter_dataset

model_name = "gpt-4o-2024-08-06"


# captioning function
def generate_caption(image_url, openai_client, prompt, temperature=1.0):
    """
    Generates a caption for an image.

    Inputs:
    - image_url (str): url of image to caption.
    - prompt (str): prompt to use for captioning.
    - temperature (float; optional): temperature setting for model, greater than 0. Defaults to 1.0; lower values are more deterministic.

    Output:
    - (str): caption for image.
    """
    response = openai_client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
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
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[],
        temperature=temperature,
        max_output_tokens=300,
        top_p=1,
        store=False,
    )

    if response.output_text is not None:
        return response.output_text
    else:
        return ""


def generate_caption_parallel(dataset, scratch_path, num_workers=4):
    """
    Generates captions for a dataset in parallel.
    TODO: this ran into an issue with the API and did not complete. Needs error handling.

    Inputs:
    - dataset (pd.DataFrame): dataframe containing image annotations and image quality.
    - openai_client (openai.OpenAI): openai client.
    """
    # get how much data we have
    total = len(dataset)
    chunk_size = math.ceil(total / num_workers)

    with Pool(num_workers) as pool:
        results = pool.map(
            generate_caption_output,
            [
                (
                    dataset[i * chunk_size : min((i + 1) * chunk_size, total)],
                    scratch_path,
                    i * chunk_size,
                )
                for i in range(num_workers)
            ],
        )

    return results


def generate_caption_output(args):
    """
    Generates a caption for an image.

    Inputs:
    - image_captioning_input (dict): dictionary containing image annotations and image quality.
    - openai_client (openai.OpenAI): openai client.
    - scratch_path (str): path to scratch folder where intermediate files will be stored.
    - start_index (int): start index of the dataset to caption.

    Output:
    - (list): list of dictionaries containing image annotations and image quality.
    """
    # unpack arguments
    image_captioning_input, scratch_path, start_index = args

    # deepclone input where labels will be
    caption_output = copy.deepcopy(image_captioning_input)

    # parameters to model
    prompt = get_prompt()

    # create scratch path if it doesn't exist
    os.makedirs(scratch_path, exist_ok=True)

    # if start index is not None, only caption the images from the start index to the end index
    if start_index is None:
        start_index = 0

    # initialize openai client
    openai_client = OpenAI()
    openai_client.api_key = os.getenv("OPENAI_API_KEY")

    for index, _ in enumerate(tqdm(image_captioning_input)):
        # get image for current annotation
        image_url = caption_output[index]["vizwiz_url"]

        # generate caption and store for output
        caption_output[index]["model_captions"] = [
            {
                "model_name": model_name,
                "caption": generate_caption(image_url, openai_client, prompt),
            }
        ]

        # save scratch file for every 100 images
        if index % 100 == 0:
            with open(
                os.path.join(
                    scratch_path,
                    f"{model_name}_caption-output_start-{start_index}_end-{start_index + index}_{datetime.now().strftime('%Y-%m-%d')}.json",
                ),
                "w",
            ) as f:
                json.dump(caption_output, f, indent=4, separators=(",", ": "))

    return caption_output


def parse_args():
    parser = argparse.ArgumentParser()

    # get the file to caption
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to the input file to caption, as JSON file. If not entered, default dataset will be used.",
    )

    # where to save file
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to the output file to save the caption output, as JSON file. If not entered, caption output will be saved to the input file's directory.",
    )

    # where to save scratch files
    parser.add_argument(
        "--scratch-path",
        type=str,
        default=None,
        help="Path to the scratch folder where intermediate files will be stored.",
    )

    # get the start and ending index of the dataset to caption
    parser.add_argument(
        "--start", type=int, default=0, help="Start index of the dataset to caption."
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index of the dataset to caption. If not entered, entire dataset will be captioned",
    )
    args = parser.parse_args()

    return args


def main():
    # get arguments
    args = parse_args()

    # load data from input file if specified
    if args.input_file is not None:
        dataset_to_caption = json.load(open(args.input_file))
    else:
        # default dataset
        print("Generating target dataset...")
        dataset_to_caption = generate_target_dataset(
            "../data/caption-dataset/annotations/train.json",
            "../data/image-quality-assessment/annotations/train.json",
        )
        dataset_to_caption = filter_dataset(pd.DataFrame.from_dict(dataset_to_caption))
        print(f"Filtered dataset of {len(dataset_to_caption)} images.")

    # get the start and ending index of the dataset to caption
    start_index = args.start
    end_index = args.end if args.end is not None else len(dataset_to_caption)

    # generate caption output
    print(f"Generating caption output for {start_index} to {end_index} images...")
    print(f"Prompt: \n {get_prompt()}")
    scratch_path = (
        args.scratch_path
        if args.scratch_path is not None
        else "../data/scratch/study-2/gpt4o-caption-output"
    )
    caption_output = generate_caption_output(
        (dataset_to_caption[start_index:end_index], scratch_path, start_index),
    )

    # TODO: fix the parallelization
    # caption_output = generate_caption_parallel(
    #     dataset_to_caption[start_index:end_index],
    #     scratch_path,
    #     num_workers=12,
    # )

    # sort caption output by image id
    caption_output = sorted(caption_output, key=lambda x: x["image_id"])

    # generate output path
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.input_file)
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{model_name}_caption-output_{len(dataset_to_caption)}-images_start-{start_index}_end-{end_index}_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.json"

    # save caption output by combining output path and filename
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(caption_output, f, indent=4, separators=(",", ": "))

    print(f"Captioning complete.Caption output saved to {output_path}")


if __name__ == "__main__":
    main()
