"""
Generates additional samples for captioned dataset.

Usage:
python generate_samples.py \
    --input-file ./coded-data/cleaned/final-image-sample_945-images_09-23-25.csv \
    --models gpt \
    --num-samples 10 \
    --start-index 0 \
    --end-index 10 \
    --greedy-response True
"""

import os
import sys
import traceback

sys.path.append("../")

import pandas as pd
import json
import gc
import argparse
import torch
from PIL import Image
import io
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import AutoModelForCausalLM
from datetime import datetime
from tqdm import tqdm

from generate_captions import (
    get_vlm_prompt,
    convert_to_png,
    get_gpt_caption,
    get_gemini_caption,
    get_llama_caption,
    get_molmo_caption,
)

load_dotenv()


def parse_args():
    """Parses the arguments from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=False)
    parser.add_argument("--num-samples", type=int, required=False, default=10)
    parser.add_argument("--start-index", type=int, required=False)
    parser.add_argument("--end-index", type=int, required=False)
    parser.add_argument("--greedy-response", type=bool, required=False, default=False)

    # specify which models as a list separated by commas
    parser.add_argument(
        "--models",
        default=[],
        nargs="+",
        required=True,
        help="List of models separated by commas (e.g., gpt, gemini, llama, molmo)",
    )

    return parser.parse_args()


def verify_models(models):
    """
    Verifies the models are valid and returns the corresponding model names. Valid models are gpt, gemini, llama, and molmo, currently.

    Args:
        models (list): List of models to verify.

    Returns:
        list: List of valid model names.
    """
    model_mapping = {
        "gpt": "gpt-4.1",
        "gemini": "gemini-2.5-flash",
        "llama": "llama-90B-4bit",
        "molmo": "molmo-72B-4bit",
    }
    if not models:
        raise ValueError("Models list is empty")
    if not all(model in model_mapping.keys() for model in models):
        raise ValueError("Invalid model specified")
    return [model_mapping[model] for model in models]


def get_additional_samples(
    image_dict,
    models,
    num_samples,
    start_index,
    end_index,
    model_settings,
    greedy_model_settings,
    greedy_response=False,
    intermediate_save=25,
):
    """
    Gets additional samples for an image.

    Args:
        image_dict (dict): The image dictionary to get additional samples for.
        models (list): The models to get additional samples for.
        num_samples (int): The number of samples to get.
        start_index (int): The start index to get samples for.
        end_index (int): The end index to get samples for.
        model_settings (dict): The model settings to use for the additional samples.
        greedy_model_settings (dict): The model settings to use for the greedy response.
        greedy_response (bool): Whether to get a greedy response. Default is False.
        intermediate_save (int): The number of images to save intermediate files for. Default is 100.

    Returns:
        dict: The image dictionary with the additional samples.
    """
    # create a new dictionary to store the additional samples
    image_dict_additional_samples = image_dict.copy()
    run_date = datetime.now().strftime("%Y-%m-%d_%H:%M")

    # get prompt
    vlm_prompt = get_vlm_prompt()

    for model_tag in models:
        model_correct_key = f"{model_tag}_correct"
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
            tqdm(image_dict_additional_samples[start_index:end_index])
        ):
            image_index += start_index
            image_url = image_info["image_url"]
            caption_name = f"{model_tag}"

            try:
                # check if a caption already exists for the model before continuing
                # Structure of output is samples: captions: { model_name: {samples: [sample1, sample2, ...], greedy_response: greedy_response}}
                if (
                    image_dict_additional_samples[image_index].get(caption_name, "")
                    != ""
                    and image_dict_additional_samples[image_index]
                    .get("captions", {})
                    .get(caption_name, {})
                    .get("samples", None)
                    is not None
                    and len(
                        image_dict_additional_samples[image_index]
                        .get("captions", {})
                        .get(caption_name, {})
                        .get("samples", [])
                    )
                    >= num_samples
                    and image_dict_additional_samples[image_index]
                    .get("captions", {})
                    .get(caption_name, {})
                    .get("greedy_response", None)
                    is not None
                ):
                    print(
                        f"{image_index} ({image_url}) already has a caption and samples for {model_tag}...skipping."
                    )
                    continue

                # generate additional samples
                image_captions = {
                    "was_correct": image_dict_additional_samples[image_index].get(
                        model_correct_key, None
                    ),
                    "samples": [],
                    "greedy_response": image_dict_additional_samples[image_index]
                    .get("captions", {})
                    .get(caption_name, {})
                    .get("greedy_response", ""),
                    "reference_caption": image_dict_additional_samples[image_index].get(
                        caption_name, ""
                    ),
                }

                # copy over all samples that are already present
                samples_left = num_samples
                if (
                    image_dict_additional_samples[image_index].get(caption_name, None)
                    is not None
                    and len(
                        image_dict_additional_samples[image_index]
                        .get("captions", {})
                        .get(caption_name, {})
                        .get("samples", [])
                    )
                    > 0
                ):
                    image_captions["samples"] = image_dict_additional_samples[
                        image_index
                    ]["captions"][caption_name]["samples"]
                    samples_left -= len(image_captions["samples"])
                    print(
                        f"Copied over {len(image_captions['samples'])} samples from existing caption for {model_tag} for image {image_dict_additional_samples[image_index]['id']} ({image_url}). Generating {samples_left} more."
                    )

                if samples_left > 0:
                    image = Image.open(io.BytesIO(convert_to_png(image_url)))
                else:
                    print(
                        f"No samples left to generate for {model_tag} for image {image_dict_additional_samples[image_index]['id']} ({image_url})...skipping."
                    )
                    image = None

                for _ in range(samples_left):
                    # run the appropriate captioning code
                    try:
                        if model_tag == "gpt-4.1":
                            image_captions["samples"].append(
                                get_gpt_caption(
                                    image_url,
                                    openai_client,
                                    model_name,
                                    vlm_prompt,
                                    **model_settings,
                                )
                            )
                        elif model_tag == "gemini-2.5-flash":
                            image_captions["samples"].append(
                                get_gemini_caption(
                                    image_url,
                                    google_client,
                                    vlm_prompt,
                                    **model_settings,
                                )
                            )
                        elif model_tag == "llama-90B-4bit":
                            image_captions["samples"].append(
                                get_llama_caption(
                                    image,
                                    model,
                                    processor,
                                    vlm_prompt,
                                    do_sample=True,
                                    **model_settings,
                                )
                            )
                        elif model_tag == "molmo-72B-4bit":
                            image_captions["samples"].append(
                                get_molmo_caption(
                                    image,
                                    model,
                                    processor,
                                    vlm_prompt,
                                    do_sample=True,
                                    **model_settings,
                                )
                            )
                    except Exception as e:
                        image_captions["samples"].append("")
                        print(
                            f"Error processing image {image_index} ({image_url}) for {model_tag} getting additional samples: {e}"
                        )

                # get greedy response
                if greedy_response:
                    # check if greedy response already exists
                    if image_captions["greedy_response"] != "":
                        print(
                            f"Greedy response already exists for image {image_dict_additional_samples[image_index]['id']} ({image_url}) for {model_tag}...skipping."
                        )
                    else:
                        # load image if not already loaded and get greedy response
                        if image is None:
                            image = Image.open(io.BytesIO(convert_to_png(image_url)))

                        try:
                            if model_tag == "gpt-4.1":
                                image_captions["greedy_response"] = get_gpt_caption(
                                    image_url,
                                    openai_client,
                                    model_name,
                                    vlm_prompt,
                                    **greedy_model_settings,
                                )
                            elif model_tag == "gemini-2.5-flash":
                                image_captions["greedy_response"] = get_gemini_caption(
                                    image_url,
                                    google_client,
                                    vlm_prompt,
                                    **greedy_model_settings,
                                )
                            elif model_tag == "llama-90B-4bit":
                                image_captions["greedy_response"] = get_llama_caption(
                                    image,
                                    model,
                                    processor,
                                    vlm_prompt,
                                    do_sample=True,
                                    **greedy_model_settings,
                                )
                            elif model_tag == "molmo-72B-4bit":
                                image_captions["greedy_response"] = get_molmo_caption(
                                    image,
                                    model,
                                    processor,
                                    vlm_prompt,
                                    do_sample=True,
                                    **greedy_model_settings,
                                )
                        except Exception as e:
                            image_captions["greedy_response"] = ""
                            print(
                                f"Error processing image {image_index} ({image_url}) for {model_tag} getting greedy response: {e}"
                            )

                # clean up any unicode
                image_captions["samples"] = [
                    sample.encode("utf-8").decode("utf-8")
                    for sample in image_captions["samples"]
                ]
                image_captions["reference_caption"] = (
                    image_captions["reference_caption"].encode("utf-8").decode("utf-8")
                )
                image_captions["greedy_response"] = (
                    image_captions["greedy_response"].encode("utf-8").decode("utf-8")
                )

                # save additional samples
                if (
                    image_dict_additional_samples[image_index].get("captions", None)
                    is None
                ):
                    image_dict_additional_samples[image_index]["captions"] = {}
                image_dict_additional_samples[image_index]["captions"][caption_name] = (
                    image_captions
                )
            except Exception as e:
                print(
                    f"Error loading image {image_index} ({image_url}) for {model_tag}: {e}"
                )
                traceback.print_exc()
                continue

            # save intermediate files every 100 captions
            if image_index != start_index and (image_index % intermediate_save) == 0:
                base_dir = f"./sampled-data/intermediate-checkpoints/{run_date}_{'-'.join(models)}"
                intermediate_output_file = f"{base_dir}/combined-sample-{start_index}-to-{image_index}_{model_tag}.json"
                print(
                    f"Saving intermediate file for {model_tag} to {intermediate_output_file} for {start_index} to {image_index}."
                )

                os.makedirs(
                    base_dir,
                    exist_ok=True,
                )
                with open(intermediate_output_file, "w") as f:
                    json.dump(
                        image_dict_additional_samples[start_index:image_index],
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

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
    return image_dict_additional_samples


def main():
    args = parse_args()

    # load data
    input_file = args.input_file
    if input_file.endswith(".csv"):
        input_data = pd.read_csv(input_file).to_dict(orient="records")
    elif input_file.endswith(".json"):
        with open(input_file, "r") as f:
            input_data = json.load(f)
    else:
        raise ValueError(f"Input file must be a CSV or JSON file. Got {input_file}")

    start_index = args.start_index if args.start_index is not None else 0
    end_index = args.end_index if args.end_index is not None else len(input_data)
    output_data_size = input_data[start_index:end_index]
    print(f"Total dataset size: {len(input_data)}")
    print(
        f"Selected {len(output_data_size)} images from {input_file} from {start_index} to {end_index}"
    )

    # get num samples and start / end index
    num_samples = args.num_samples
    greedy_response = args.greedy_response

    # get models
    models = verify_models(args.models)
    print(f"Using models: {models}")

    # get output file
    output_file = (
        args.output_file
        if args.output_file is not None
        else f"./sampled-data/sampled-data_{start_index}-to-{end_index}_{'-'.join(models)}_{len(output_data_size)}-images_{num_samples}-samples_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.json"
    )
    print(f"Output will be saved to: {output_file}")

    # generate samples
    model_settings = {
        "temperature": 1.0,
        "top_p": 0.95,
    }
    greedy_model_settings = {
        "temperature": 0.0,
        "top_p": 0.95,
    }
    print(f"Using model settings: {model_settings}")
    print(f"Using greedy model settings: {greedy_model_settings}")

    data_with_samples = get_additional_samples(
        input_data,
        models,
        num_samples,
        start_index,
        end_index,
        model_settings,
        greedy_model_settings,
        greedy_response=greedy_response,
    )
    print(f"Generated {len(data_with_samples)} samples")

    # save data with samples
    os.makedirs(
        "./sampled-data/",
        exist_ok=True,
    )
    with open(output_file, "w") as f:
        json.dump(
            data_with_samples[start_index:end_index], f, indent=2, ensure_ascii=False
        )
    print(f"Saved {len(data_with_samples)} samples to {output_file}")


if __name__ == "__main__":
    main()
