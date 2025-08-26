# ------------------- Data Processing -------------------
import sys

sys.path.append("../")

import pandas as pd
import os
import json
import gc
from datetime import datetime

# ------------------- Progress Bar -------------------
from tqdm import tqdm

# ------------------- Environment Variables -------------------
from dotenv import load_dotenv

# ------------------- Image Processing -------------------
import requests
from PIL import Image
from io import BytesIO
import base64
import io

# VLMs
import torch
from openai import OpenAI
from google import genai
from google.genai import types

from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import AutoModelForCausalLM

from scripts.molmo_captioner import generate_caption as get_molmo_caption
from scripts.llama_captioner import generate_caption as get_llama_caption

# ------------------- Setup -------------------
load_dotenv()

# ------------------- Load Data -------------------
captioned_file = (
    "./input-data/combined-image-input_1997_partially-completed_2025-08-25_16-30.json"
)

combined_sample_dict = None
if captioned_file and os.path.isfile(captioned_file):
    with open(captioned_file, "r") as f:
        combined_sample_dict = json.load(f)

# show the data we're working with
print(f"Number of samples: {len(combined_sample_dict)}")


# ------------------- Image Processing -------------------
def remove_transparency(im, bg_colour=(255, 255, 255)):
    """
    Remove transparency from an image.

    Args:
        im (PIL.Image.Image): Image to remove transparency from.
        bg_colour (tuple, optional): Background color to use for the transparent areas. Defaults to (255, 255, 255).

    Returns:
        PIL.Image.Image: Image with transparency removed.
    """
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert("RGBA").split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im


def convert_to_base64(image_url):
    """
    Convert an image, specified by its url, to a PNG and return the base64 encoded string.

    Args:
        image_url (str): URL of the image to convert.

    Returns:
        str: Base64 encoded string of the image.
    """
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # remove transparency
    image = remove_transparency(image)

    with BytesIO() as f:
        image.save(f, format="PNG")
        f.seek(0)

        return base64.b64encode(f.read()).decode("utf-8")


def convert_to_png(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # remove transparency
    image = remove_transparency(image)

    with BytesIO() as f:
        image.save(f, format="PNG")
        return f.getvalue()


# ------------------- Caption Generation -------------------
def get_gpt_caption(
    image_url, openai_client, model_name, prompt, temperature=1.0, top_p=1.0, **kwargs
):
    # convert image_url to base54
    image_b64 = convert_to_base64(image_url)

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
                        "image_url": f"data:image/png;base64,{image_b64}",
                        "detail": "high",
                    }
                ],
            },
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[],
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=500,
        store=False,
    )

    if response.output_text is not None:
        return response.output_text
    else:
        return ""


def get_gemini_caption(
    image_url, client, prompt, temperature=1.0, top_p=0.95, **kwargs
):
    # convert image to bytes
    image_b64 = convert_to_base64(image_url)

    # get caption
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        mime_type="image/png",
                        data=base64.b64decode(image_b64),
                    )
                ],
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=500,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
            ),
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            system_instruction=[types.Part.from_text(text=prompt)],
        ),
    )

    if response.text is not None:
        return response.text
    else:
        return ""


# ------------------- Model Settings -------------------
models = [
    "gpt-4.1",
    "gemini-2.5-flash",
    "llama-90B-4bit",
    "molmo-72B-4bit",
]

MODEL_SETTINGS = dict(temperature=1.0, top_p=0.95)

START_IDX = 0
END_IDX = len(combined_sample_dict)

VLM_PROMPT = (
    "You are a helpful assistant who identifies products in images for blind and low-vision individuals. Identify the product in the image while following these guidelines:\n"
    "1: Identify crucial features about the product, including:\n"
    "-- Object type (can, bag, plastic container, etc.) \n"
    "-- Product type (prepared or frozen meal, seasoning mix, soda, coffee) \n"
    "-- Brand (Heinz, Coca-Cola, Starbucks, etc.) \n"
    "-- Variety (specific flavors, sizes, count of items, etc.) \n"
    "-- Visual features (color, shape, size, etc.) \n"
    "2: Use clear, direct, and objective language. Do not use vague adjectives like 'large' or 'small', or vague adverbs like 'prominently' or 'clearly'.\n"
    "3: DO NOT mention camera artifacts (e.g., blur) or if an object is partially visible.\n"
    "4: DO NOT use introductory phrases (e.g., 'The image shows', 'The object is', 'The primary object is').\n\n"
    "Output only the final description of the product."
)

print(f"Models: {models}")
print(f"Model settings: {MODEL_SETTINGS}")
print(
    f"Images in dataset: {len(combined_sample_dict)} | Start Index: {START_IDX}, END_IDX: {END_IDX}, Images to Caption: {END_IDX - START_IDX}"
)
print(f"Prompt: \n{VLM_PROMPT}")

# ------------------- Caption Generation -------------------
run_date = datetime.now().strftime("%Y-%m-%d_%H:%M")
for model_tag in models:
    # load relevant model
    if model_tag == "gpt-4.1":
        openai_client = OpenAI()
        openai_client.api_key = os.getenv("OPENAI_API_KEY")
        model_name = "gpt-4.1-2025-04-14"

    elif model_tag == "gemini-2.5-flash":
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.
        google_client = genai.Client()
        model_name = "gemini-2.5-flash"

    elif model_tag == "llama-90B-4bit":
        model_name = "Llama-3.2-90B-Vision-Instruct-bnb-4bit"
        model_id = "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit"
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)

        # print model properties
        print("Model ID: ", model_id)
        print("Device: ", model.device)
        print("Dtype: ", model.dtype)

    elif model_tag == "molmo-72B-4bit":
        # For 2 x 24 GB. 1 x 48 GB or more *should* work on just 1 GPU, but I've ran out of memory
        device_map = {
            "model.vision_backbone": 0,
            "model.transformer.wte": 0,
            "model.transformer.ln_f": 0,
            "model.transformer.ff_out": 1,
        }

        # For 2 x 24 GB, this works for *only* 38 or 39. Any higher or lower and it'll either only work for 1 token of output or fail completely.
        switch_point = 38  # layer index to switch to second GPU
        device_map |= {
            f"model.transformer.blocks.{i}": 0 for i in range(0, switch_point)
        }
        device_map |= {
            f"model.transformer.blocks.{i}": 1 for i in range(switch_point, 80)
        }

        # model_name = "SeanScripts/Molmo-72B-0924-nf4"
        model_name = "kgarg0/Molmo-72B-0924-nf4-fixed"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,
            device_map=device_map,
            trust_remote_code=True,  # Required for Molmo at the moment.
        )
        model.model.vision_backbone.float()  # vision backbone needs to be in FP32 for this

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,  # Required for Molmo at the moment.
        )

        # print model properties
        print("Model ID: ", model_name)
        print("Device: ", model.device)
        print("Dtype: ", model.dtype)

    print(f"Generating Captions for {model_tag} using {model_name}")

    for image_index, image_info in enumerate(
        tqdm(combined_sample_dict[START_IDX:END_IDX])
    ):
        image_index += START_IDX
        image_url = image_info["image_url"]
        caption_name = f"{model_tag}"

        # check if a caption already exists for the model before continuing
        if (
            caption_name in combined_sample_dict[image_index]
            and combined_sample_dict[image_index][caption_name] != ""
        ):
            print(
                f"{image_index} ({image_url}) already has a caption for {model_tag}...skipping."
            )
            continue

        # run the appropriate captioning code
        try:
            image = Image.open(io.BytesIO(convert_to_png(image_url)))

            if model_tag == "gpt-4.1":
                combined_sample_dict[image_index][caption_name] = get_gpt_caption(
                    image_url, openai_client, model_name, VLM_PROMPT, **MODEL_SETTINGS
                )
            elif model_tag == "gemini-2.5-flash":
                combined_sample_dict[image_index][caption_name] = get_gemini_caption(
                    image_url, google_client, VLM_PROMPT, **MODEL_SETTINGS
                )
            elif model_tag == "llama-90B-4bit":
                combined_sample_dict[image_index][caption_name] = get_llama_caption(
                    image,
                    model,
                    processor,
                    VLM_PROMPT,
                    do_sample=True,
                    **MODEL_SETTINGS,
                )
            elif model_tag == "molmo-72B-4bit":
                combined_sample_dict[image_index][caption_name] = get_molmo_caption(
                    image,
                    model,
                    processor,
                    VLM_PROMPT,
                    do_sample=True,
                    **MODEL_SETTINGS,
                )
        except Exception as e:
            print(
                f"Error processing image {image_index} ({image_url}) for {model_tag}: {e}"
            )

            # empty caption if error
            combined_sample_dict[image_index][caption_name] = ""

        # save intermediate files every 100 captions
        if image_index != START_IDX and (image_index % 100) == 0:
            os.makedirs(
                f"./captioned-data/intermediate-checkpoints/{run_date}/", exist_ok=True
            )
            intermediate_output_file = f"./captioned-data/intermediate-checkpoints/{run_date}/combined-sample-{START_IDX}-to-{image_index}_{model_tag}.json"
            print(
                f"Saving intermediate file for {model_tag} to {intermediate_output_file} for {START_IDX} to {image_index}."
            )

            with open(intermediate_output_file, "w") as f:
                json.dump(combined_sample_dict[START_IDX:image_index], f, indent=2)

    # clean-up
    if model_tag == "gpt-4.1":
        del openai_client
    elif model_tag == "gemini-2.5-flash":
        del google_client
    elif model_tag == "llama-90B-4bit":
        # clear cache and model objects
        del model_id, model, processor
        torch.cuda.empty_cache()
        gc.collect()
    elif model_tag == "molmo-72B-4bit":
        # clear cache and model objects
        del device_map, switch_point, model, processor
        torch.cuda.empty_cache()
        gc.collect()
    del model_name
    print("-" * 80)

# ------------------- Save Data -------------------
# Save JSON
os.makedirs("./captioned-data/", exist_ok=True)
output_file_json_name = f"./captioned-data/captioned-data-all-models_{START_IDX}-to-{END_IDX}_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.json"
with open(output_file_json_name, "w") as f:
    json.dump(combined_sample_dict[START_IDX:END_IDX], f, indent=2)
print(f"Captioned data saved as JSON to: {output_file_json_name}")

# Save CSV
column_order = {
    "annotation_info": [
        "id",
        "orig_id",
        "file_name",
        "image_url",
        "image_preview",
        "type",
        "human_captions",
        "expert_caption",
        "orig annotator",
        "orig annotation notes",
        "unable_to_verify",
        "double code notes",
        "double verified",
        "annotator",
        "annotation notes",
        "object",
        "product",
        "brand",
        "variety",
        "double annotator",
        "double annotation",
    ],
    "image_quality": [
        "text_detected",
        "curved label",
        "text panel",
        "unrecognizable",
        "framing",
        "blur",
        "obstruction",
        "rotation",
        "too dark",
        "too bright",
        "other",
        "unrecognizable_orig",
        "framing_orig",
        "blur_orig",
        "obstruction_orig",
        "rotation_orig",
        "too_dark_orig",
        "too_bright_orig",
        "other_orig",
        "no_issue_orig",
    ],
}

# arrange columns
model_col_order = []
for model_tag in models:
    model_col_order.append(f"{model_tag}")

final_column_order = (
    column_order["annotation_info"] + model_col_order + column_order["image_quality"]
)

# output formatted csv
output_df = pd.DataFrame(combined_sample_dict[START_IDX:END_IDX])
output_df = output_df[final_column_order]

# save
output_file_csv_name = f"./captioned-data/captioned-data-all-models_{START_IDX}-to-{END_IDX}_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.csv"
output_df.to_csv(
    output_file_csv_name,
    index=False,
)
print(f"Captioned data saved as CSV to: {output_file_csv_name}")
