# Libraries
import argparse
import json
import pandas as pd
import os
import copy
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


def generate_caption_output(
    image_captioning_input, openai_client, scratch_path, start_index
):
    """
    Generates a caption for an image.

    Inputs:
    - image_captioning_input (pd.DataFrame): dataframe containing image annotations and image quality.
    - openai_client (openai.OpenAI): openai client.
    - scratch_path (str): path to scratch folder where intermediate files will be stored.
    - start_index (int): start index of the dataset to caption.

    Output:
    - (list): list of dictionaries containing image annotations and image quality.
    """
    # deepclone input where labels will be
    caption_output = copy.deepcopy(image_captioning_input)

    # parameters to model
    prompt = get_prompt()

    # create scratch path if it doesn't exist
    os.makedirs(scratch_path, exist_ok=True)

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
                    f"{model_name}_caption-output_start-{start_index}_end-{start_index + index}.json",
                ),
                "w",
            ) as f:
                json.dump(caption_output, f, indent=4, separators=(",", ": "))

    return caption_output


def main():
    # load data
    print("Generating target dataset...")
    dataset_to_caption = generate_target_dataset(
        "../data/caption-dataset/annotations/train.json",
        "../data/image-quality-assessment/annotations/train.json",
    )
    dataset_to_caption = filter_dataset(pd.DataFrame.from_dict(dataset_to_caption))
    print(f"Filtered dataset of {len(dataset_to_caption)} images.")

    # get the start and ending index of the dataset to caption
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    start_index = args.start
    end_index = args.end if args.end is not None else len(dataset_to_caption)

    # initialize openai client
    openai_client = OpenAI()
    openai_client.api_key = os.getenv("OPENAI_API_KEY")

    # generate caption output
    print(f"Generating caption output for {start_index} to {end_index} images...")
    caption_output = generate_caption_output(
        dataset_to_caption[start_index:end_index],
        openai_client,
        "../data/scratch/study-2/gpt4o-caption-output",
        start_index,
    )

    # save caption output
    output_path = "../data/study-2-output/labeled-data/gpt4o-caption-output"
    os.makedirs(output_path, exist_ok=True)
    with open(
        f"{output_path}/{model_name}_caption-output_{len(caption_output)}-images_start-{start_index}_end-{end_index}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json",
        "w",
    ) as f:
        json.dump(caption_output, f, indent=4, separators=(",", ": "))


if __name__ == "__main__":
    main()
