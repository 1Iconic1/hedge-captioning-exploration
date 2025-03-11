# Libraries
import json
import csv
import pandas as pd
import os
import re
import copy

import requests
from PIL import Image

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch

from tqdm import tqdm

# setup pytorch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for multi-GPU systems, force single GPU
if torch.cuda.is_available():
    device_map = "cuda:0"  # force single, first GPU
    device_type = "cuda"
elif torch.backends.mps.is_available():
    device_map = "auto"
    device_type = "mps"
else:
    device_map = "auto"
    device_type = "cpu"

print(f"Using device: {device_type}")


# load model
model_id = "allenai/Molmo-7B-O-0924"
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
)

# print model properties
print("Model ID: ", model_id)
print("Device: ", model.device)
print("Dtype: ", model.dtype)


# captioning function
def generate_caption(
    image_object, model, processor, prompt, temperature=1.0, do_sample=False
):
    """
    Generates a caption for an image.

    Inputs:
    - image_object (pil Image): image to caption.
    - model (torch model): loaded model to use for captioning.
    - processor (torch processor): loaded processor for pre-processing inputs.
    - temperature (float; optional): temperature setting for model, greater than 0. Defaults to 1.0; lower values are more deterministic.
    - do_sample (boolean; optional): whether model should sample probabilities. Defaults to False -- greedy decoding.

    Output:
    - (str): caption for image.
    """
    # process the image and text
    inputs = processor.process(
        images=[image_object],
        text=prompt,
    )

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = ""
    with torch.autocast(device_type=device_type, enabled=True, dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
            use_cache=False,
            temperature=temperature,
            do_sample=do_sample,
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        output = generated_text.strip()

    return output


def generate_target_dataset(caption_dataset_filename, image_quality_dataset_filename):
    """
    Generates a target dataset for captioning based on VizWiz's image captioning dataset and image quality assessment dataset.

    Inputs:
    - caption_dataset_filename (str): path to caption dataset.
    - image_quality_dataset_filename (str): path to image quality dataset.

    Output:
    - (pd.DataFrame): dataframe containing image annotations and image quality.
    """
    # get images and annotations in one dataframe
    image_annotation_df = None
    with open(caption_dataset_filename) as f:
        # load caption dataset
        caption_dataset_json = json.load(f)

        # combine image files and annotations
        images_df = pd.DataFrame.from_dict(caption_dataset_json["images"])
        annotations_df = pd.DataFrame.from_dict(caption_dataset_json["annotations"])
        grouped_annotations = (
            annotations_df.groupby(["image_id"]).agg(tuple).map(list).reset_index()
        )
        image_annotation_df = images_df.merge(
            grouped_annotations[["image_id", "caption", "is_precanned", "is_rejected"]],
            left_on="id",
            right_on="image_id",
        )

        # vizwiz_url is broken, so fix with https://vizwiz.cs.colorado.edu/*
        image_annotation_df["vizwiz_url"] = image_annotation_df["vizwiz_url"].apply(
            lambda x: x.replace(
                "https://ivc.ischool.utexas.edu/", "https://vizwiz.cs.colorado.edu/"
            )
        )

    # get image quality
    with open(image_quality_dataset_filename) as f:
        # load image quality annotation dataset
        image_quality_dataset_json = json.load(f)
        image_quality_df = pd.DataFrame.from_dict(image_quality_dataset_json)

        # expand object of flaws into individual columns and rename
        image_quality_df = pd.concat(
            [
                image_quality_df.drop(["flaws"], axis=1),
                pd.json_normalize(image_quality_df["flaws"]),
            ],
            axis=1,
        )
        image_quality_df.rename(
            columns={
                "FRM": "framing",
                "BLR": "blur",
                "DRK": "too dark",
                "BRT": "too bright",
                "OBS": "obstruction",
                "OTH": "other",
                "NON": "no issue",
                "ROT": "rotation",
                "caption": "human_captions",
            },
            inplace=True,
        )

    # combine image and quality datasets together
    image_captioning_input = image_annotation_df.merge(
        image_quality_df, left_on="file_name", right_on="image"
    ).drop(["image"], axis=1)

    # convert image_captioning_input to a list of dictionaries
    image_captioning_input = image_captioning_input.to_dict(orient="records")
    return image_captioning_input


def generate_caption_output(image_captioning_input, image_folder, scratch_path):
    """
    Generates a caption for an image.

    Inputs:
    - image_captioning_input (pd.DataFrame): dataframe containing image annotations and image quality.
    - image_folder (str): path to image folder.
    - scratch_path (str): path to scratch folder where intermediate files will be stored.

    Output:
    - (list): list of dictionaries containing image annotations and image quality.
    """
    # deepclone input where labels will be
    caption_output = copy.deepcopy(image_captioning_input)

    # parameters to model
    prompt = "You are a program designed to help blind and low-vision users understand images. When asked about the image, generate accessible image description that includes key visual and contextual details of the image for blind and low-vision people. Focus on the following principles: Clarity and Conciseness: Use simple, straightforward language to describe the main subjects and their relationships.; Relevance: Highlight only essential visual elements that contribute to understanding the image or its purpose.; Context: Provide contextual information when necessary, such as emotional tone, setting, or action. Avoid assumptions or subjective interpretations.; Specificity: Include important details like colors, shapes, textures, or text visible in the image, if relevant. Avoid overly general terms or unnecessary details. Once you generate your caption, shorten it to a succinct, single sentence. Output only the final sentence. Can you please tell me what is in this image?"

    # create scratch path if it doesn't exist
    os.makedirs(scratch_path, exist_ok=True)

    for index, row in enumerate(tqdm(image_captioning_input)):
        # get image for current annotation
        image_file = os.path.join(image_folder, caption_output[index]["file_name"])
        image = Image.open(image_file)

        # generate caption and store for output
        caption_output[index]["model_caption"] = generate_caption(
            image, model, processor, prompt
        )

        # save scratch file for every 100 images
        if index % 100 == 0:
            with open(
                os.path.join(scratch_path, f"caption_output_{index}.json"), "w"
            ) as f:
                json.dump(caption_output, f, indent=4, separators=(",", ": "))

    return caption_output


def main():
    print("Generating target dataset...")
    dataset_to_caption = generate_target_dataset(
        "../data/caption-dataset/annotations/train.json",
        "../data/image-quality-assessment/annotations/train.json",
    )

    print("Generating caption output...")
    caption_output = generate_caption_output(
        dataset_to_caption,
        "../data/caption-dataset/train",
        "../data/scratch/vizwiz-image-captions",
    )

    # save caption output
    os.makedirs("../data/labeled-data/vizwiz-image-captions", exist_ok=True)
    with open(
        f"../data/labeled-data/vizwiz-image-captions/caption_output_{len(caption_output)}.json",
        "w",
    ) as f:
        json.dump(caption_output, f, indent=4, separators=(",", ": "))


if __name__ == "__main__":
    main()
