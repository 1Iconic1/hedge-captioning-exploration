"""
Usage
For high-quality images:
python molmo_captioner.py \
    --input-file "../data/study-2-input/high-quality-image_iq-1_text-no-text.json" \
    --output-dir "../data/study-2-output/labeled-data/high-quality-images/molmo-caption-output" \
    --scratch-path "../data/scratch/study-2/molmo-caption-output" \
    --start 0 \
    --end 10
"""

# Libraries
import argparse
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
    torch_dtype = "auto"
elif torch.backends.mps.is_available():
    device_map = "auto"
    device_type = "mps"
    torch_dtype = torch.bfloat16
else:
    device_map = "auto"
    device_type = "cpu"
    torch_dtype = torch.bfloat16
print(f"Using device: {device_type}")


# load model
model_name = "Molmo-7B-O-0924"
model_id = "allenai/Molmo-7B-O-0924"
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    device_map=device_map,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
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
    def generate_output():
        return model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=300, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
            use_cache=True,
            temperature=temperature,
            do_sample=do_sample,
        )

    output = None
    if device_type == "cuda":
        output = generate_output()
    else:
        with torch.autocast(
            device_type=device_type, enabled=True, dtype=torch.bfloat16
        ):
            output = generate_output()

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
        caption_output[index]["model_captions"] = [
            {
                "model_name": model_name,
                "caption": generate_caption(image, model, processor, prompt),
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

    # generate output
    print(f"Generating caption output for {start_index} to {end_index} images...")
    print(f"Prompt: \n {get_prompt()}")
    scratch_path = (
        args.scratch_path
        if args.scratch_path is not None
        else "../data/scratch/study-2/molmo-caption-output"
    )
    caption_output = generate_caption_output(
        dataset_to_caption[start_index:end_index],
        "../data/caption-dataset/train",
        scratch_path,
        start_index,
    )

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
