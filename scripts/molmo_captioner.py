# Libraries
import json
import pandas as pd
import os
import copy
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
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
model_id = "allenai/Molmo-7B-O-0924"
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map=device_map,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="auto",
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

    # generate output; maximum 300 new tokens; stop generation when <|endoftext|> is generated
    output = ""
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=300, stop_strings="<|endoftext|>"),
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

    return generated_text.strip()


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
    dataset_to_caption = filter_dataset(pd.DataFrame.from_dict(dataset_to_caption))
    print(f"Filtered dataset of {len(dataset_to_caption)} images.")

    print("Generating caption output...")
    caption_output = generate_caption_output(
        dataset_to_caption,
        "../data/caption-dataset/train",
        "../data/scratch/study-2/molmo-caption-output",
    )

    # save caption output
    output_path = "../data/study-2-output/labeled-data/molmo-caption-output"
    os.makedirs(output_path, exist_ok=True)
    with open(
        f"{output_path}/caption_output_{len(caption_output)}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json",
        "w",
    ) as f:
        json.dump(caption_output, f, indent=4, separators=(",", ": "))


if __name__ == "__main__":
    main()
