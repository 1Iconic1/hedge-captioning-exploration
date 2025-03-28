# Libraries
import argparse
import json
import pandas as pd
import os
import copy
from datetime import datetime

import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, GenerationConfig
from PIL import Image
from tqdm import tqdm

# custom imports
from constants import get_prompt
from data_loader import generate_target_dataset, filter_dataset


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
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
)
processor = AutoProcessor.from_pretrained(model_id)

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
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
            ],
        },
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image_object, input_text, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)

    # generate output; maximum 300 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate(
        **inputs,
        generation_config=GenerationConfig(
            max_new_tokens=300,
            stop_strings="<|endoftext|>",
            use_cache=True,
            temperature=temperature,
            do_sample=do_sample,
        ),
        tokenizer=processor.tokenizer,
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )

    return generated_text.strip()


def generate_caption_output(
    image_captioning_input, image_folder, scratch_path, start_index
):
    """
    Generates a caption for an image.

    Inputs:
    - image_captioning_input (pd.DataFrame): dataframe containing image annotations and image quality.
    - image_folder (str): path to image folder.
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
                os.path.join(
                    scratch_path,
                    f"caption_output_start-{start_index}_end-{start_index + index}.json",
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

    print(f"Generating caption output for {start_index} to {end_index} images...")
    print(f"Prompt: \n {get_prompt()}")
    caption_output = generate_caption_output(
        dataset_to_caption[start_index:end_index],
        "../data/caption-dataset/train",
        "../data/scratch/study-2/llama-caption-output",
        start_index,
    )

    # save caption output
    output_path = "../data/study-2-output/labeled-data/llama-caption-output"
    os.makedirs(output_path, exist_ok=True)
    with open(
        f"{output_path}/caption_output_{len(caption_output)}-images_start-{start_index}_end-{end_index}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json",
        "w",
    ) as f:
        json.dump(caption_output, f, indent=4, separators=(",", ": "))


if __name__ == "__main__":
    main()
