"""
Runs all evaluation metrics on the captions and outputs a json file with the results.

Usage:
Example with no evaluations:
python evaluate_captions.py \
    --input ../../data/study-2-output/labeled-data/combined-caption-output/combined-caption-output_7304-images_2025-03-29_21:40:00.json \
    --image-folder ../../data/caption-dataset/train \
    --output-dir ../../data/study-2-output/labeled-data/evaluation-results \
    --start 0 \
    --end 10

Example with some evaluations computed:
python evaluate_captions.py \
    --input ../../data/study-2-output/labeled-data/evaluation-results/evaluation_results_5432-images_2025-04-03_11:27_fixed.json \
    --image-folder ../../data/caption-dataset/train \
    --output-dir ../../data/study-2-output/labeled-data/evaluation-results \
    --start 0 \
    --end 10
"""

import argparse
import os
import json
import copy
from datetime import datetime
import time

import evaluate
import torch
from bert_score import score
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.cider.cider import Cider

import clip
import numpy as np
from clipscore import get_clip_score, get_refonlyclipscore, extract_all_images

# setup pytorch for BERTScore and CLIP
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for multi-GPU systems, force single GPU
if torch.cuda.is_available():
    device_type = "cuda:0"  # force single, first GPU
elif torch.backends.mps.is_available():
    device_type = "mps"
else:
    device_type = "cpu"
print(f"Using device: {device_type}")


def load_data(input_file):
    """
    Load the data from the input file

    Args:
        input_file (str): path to the input file

    Returns:
        list of dict: list of dictionaries containing captioning data
    """
    with open(input_file, "r") as f:
        data = json.load(f)
    return data


def sort_data(data):
    """
    Sort the data by image_id.

    Args:
        data (list of dict): list of dictionaries containing captioning data

    Returns:
        list of dict: list of dictionaries containing captioning data sorted by image_id
    """
    return sorted(data, key=lambda x: x["image_id"])


def construct_evaluation_input(dataset, image_folder):
    """
    Constructs an array of references, candidate caption, and image file (for CLIP).
    Used for batch evaluation of each metric.

    Args:
        dataset (list of dict): list of dictionaries containing captioning data. Each dictionary must contain
            - "human_captions" (list of dict): list of dictionaries containing human captions
            - "model_captions" (list of dict): list of dictionaries containing model captions
            - "image_file" (str): path to the image file

    Returns:
        (dict of list): candidate captions for each model. model name is the key.
            len(candidates[model]) == len(dataset)
        (list): references for each image. each element is a list of strings.
            len(references) == len(dataset)
        (list): image files for each image. each element is a string.
            len(image_files) == len(dataset)
    """
    candidates = {}
    references = []
    image_files = []
    for image in dataset:
        # collect references for the current image
        curr_references = [
            caption["caption"]
            for caption in image["human_captions"]
            if caption["caption"]
            != "Quality issues are too severe to recognize visual content."
        ]
        references.append(curr_references)

        # get captions for each model for the current image
        for model in image["model_captions"]:
            curr_model_name = model["model_name"]

            # compute scores for current model
            curr_candidate = model["caption"]
            if curr_model_name not in candidates:
                candidates[curr_model_name] = [curr_candidate]
            else:
                candidates[curr_model_name].append(curr_candidate)

        # construct list of files for Clip
        # check if the image file exists
        image_file_name = image["file_name"]
        if os.path.exists(os.path.join(image_folder, image_file_name)):
            image_files.append(os.path.join(image_folder, image_file_name))
        else:
            print(f"Image file {image_file_name} does not exist.")
            image_files.append("")

    return candidates, references, image_files


def execute_bleu(candidates, references, order):
    """
    Execute BLEU evaluation.

    Args:
        candidates (list of str): list of candidate captions
        references (list of list of str): list of references for each candidate caption.
            len(references) == len(candidates)
        order (int): order of BLEU to compute

    Returns:
        list of dict: list of dictionaries containing BLEU scores for each image.
    """
    output = []
    bleu = evaluate.load("bleu")
    for index in range(len(candidates)):
        candidate = candidates[index]
        reference = references[index]
        output.append(
            bleu.compute(
                predictions=[candidate],
                references=[reference],
                max_order=order,
            )
        )
    return output


def execute_meteor(candidates, references):
    """
    Execute METEOR evaluation.

    Args:
        candidates (list of str): list of candidate captions
        references (list of list of str): list of references for each candidate caption.
            len(references) == len(candidates)
        order (int): order of METEOR to compute

    Returns:
        list of dict: list of dictionaries containing METEOR scores for each image.
    """
    output = []
    meteor = evaluate.load("meteor")
    for index in range(len(candidates)):
        candidate = candidates[index]
        reference = references[index]
        output.append(
            meteor.compute(
                predictions=[candidate],
                references=[reference],
            )
        )
    return output


def execute_rouge(candidates, references):
    """
    Execute ROUGE evaluation.

    Args:
        candidates (list of str): list of candidate captions
        references (list of list of str): list of references for each candidate caption.
            len(references) == len(candidates)
        order (int): order of ROGUE to compute

    Returns:
        list of dict: list of dictionaries containing ROUGE scores for each image.
    """
    output = []
    rouge = evaluate.load("rouge")
    for index in range(len(candidates)):
        candidate = candidates[index]
        reference = references[index]
        output.append(
            rouge.compute(
                predictions=[candidate],
                references=[reference],
            )
        )
    return output


def execute_bertscore(
    candidates,
    references,
    model_type="microsoft/deberta-xlarge-mnli",
    lang="en",
    rescale_with_baseline=False,
):
    """
    Execute BERTScore evaluation.

    Args:
        candidates (list of str): list of candidate captions
        references (list of list of str): list of references for each candidate caption.
            len(references) == len(candidates)
        model_type (str): type of model to use for BERTScore evaluation
        lang (str): language of the captions for BERTScore evaluation
        rescale_with_baseline (bool): whether to rescale the scores with a baseline

    Returns:
        list of dict: list of dictionaries containing BERTScore (precision, recall, F1) for each image.
    """
    P_lst, R_lst, F1_lst = score(
        candidates,
        references,
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
    )
    output = []
    for index in range(len(candidates)):
        output.append(
            {
                "precision": float(P_lst[index]),
                "recall": float(R_lst[index]),
                "f1": float(F1_lst[index]),
            }
        )
    return output


def tokenize(refs, cands, no_op=False):
    """
    Used for CIDEr and SPICE evaluation. From: https://github.com/jmhessel/clipscore/blob/main/generation_eval_utils.py.

    Args:
        refs (_type_): _description_
        cands (_type_): _description_
        no_op (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # no_op is a debug option to see how significantly not using the PTB tokenizer
    # affects things
    tokenizer = PTBTokenizer()

    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}

    else:
        refs = {
            idx: [{"caption": r} for r in c_refs] for idx, c_refs in enumerate(refs)
        }
        cands = {idx: [{"caption": c}] for idx, c in enumerate(cands)}

        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)

    return refs, cands


def execute_cider(candidates, references):
    """
    Execute CIDEr evaluation.

    Args:
        candidates (list of str): list of candidate captions
        references (list of list of str): list of references for each candidate caption.
            len(references) == len(candidates)

    Returns:
        tuple: average CIDEr score and scores for each image.
    """
    scorer = Cider()
    refs, cands = tokenize(references, candidates)
    average_score, scores = scorer.compute_score(refs, cands)
    return average_score, scores


def execute_spice(candidates, references):
    """
    Execute SPICE evaluation.

    Args:
        candidates (list of str): list of candidate captions
        references (list of list of str): list of references for each candidate caption.
            len(references) == len(candidates)

    Returns:
        tuple: average SPICE score and scores for each image.
    """
    scorer = Spice()
    refs, cands = tokenize(references, candidates)
    average_score, scores = scorer.compute_score(refs, cands)
    return average_score, scores


def execute_clipscore(candidates, image_files):
    """
    Execute CLIPScore evaluation.

    Args:
        candidates (list of str): list of candidate captions
        image_files (list of str): list of image files

    Returns:
        list of dict: list of dictionaries containing CLIPScore scores for each image.
    """
    model, transform = clip.load("ViT-B/32", device=device_type, jit=False)
    model.eval()
    image_feats = extract_all_images(
        image_files, model, device_type, batch_size=64, num_workers=1
    )

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, candidates, device_type
    )

    return [{"score": float(x)} for x in per_instance_image_text]


def execute_clipscore_ref(candidates, references, image_files):
    """
    Execute CLIPScore with References evaluation.

    Args:
        candidates (list of str): list of candidate captions
        references (list of list of str): list of references for each candidate caption.
            len(references) == len(candidates)
        image_files (list of str): list of image files

    Returns:
        list of dict: list of dictionaries containing CLIPScore with References scores for each image.
    """
    model, transform = clip.load("ViT-B/32", device=device_type, jit=False)
    model.eval()
    image_feats = extract_all_images(
        image_files, model, device_type, batch_size=64, num_workers=1
    )

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, candidates, device_type
    )

    # get text-text clipscore
    _, per_instance_text_text = get_refonlyclipscore(
        model, references, candidate_feats, device_type
    )

    # F-score
    refclipscores = (
        2
        * per_instance_image_text
        * per_instance_text_text
        / (per_instance_image_text + per_instance_text_text)
    )

    return [{"score": float(x)} for x in refclipscores]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input JSON file path.")
    parser.add_argument("--image-folder", type=str, help="Image folder path.")
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for JSON file."
    )
    parser.add_argument(
        "--start", type=int, help="Start index of the data to evaluate."
    )
    parser.add_argument("--end", type=int, help="End index of the data to evaluate.")
    return parser.parse_args()


def check_if_evaluated(data, key):
    """
    Check if the data has already been evaluated one a measure. Key corresponds to an evaluation metric. For each image:
    {
        ...
        key: {
            "model": {
                "scores": {...}
            },
            ...
        }
        ...
    }

    Args:
        data (list of dict): list of dictionaries containing captioning data
        key (str): key of the evaluation metric

    Returns:
        (list of evaluation dicts or None): list of dictionaries containing evaluation data, if present for all images in data or None if not
    """
    output = []

    for image in data:
        if "evaluation" not in image:
            return None
        if key in image["evaluation"]:
            output.append(image["evaluation"][key])
        else:
            return None

    # at this point, we're only returning a list of evaluation dicts
    # if all have an evaluation for the given key
    return output


def main():
    # get the arguments
    args = parse_args()

    # load the data from the files
    data = load_data(args.input)

    # sort the data by image_id since order matters for matching metrics up
    data = sort_data(data)

    # get the subset of data to evaluate
    start_idx = args.start if args.start else 0
    end_idx = args.end if args.end else len(data)
    data = data[start_idx:end_idx]

    # construct the evaluation input
    candidates, references, image_files = construct_evaluation_input(
        data, args.image_folder
    )

    # evaluate the captions
    print(f"Evaluating data from {start_idx} to {end_idx} (total images: {len(data)}).")
    """
    By the end of this loop, we want an array of dictionaries with an evalutation object.
    This should be the same length as the data object.
    Each object in this list gets attached to the original data object in evalutation: {...}
    [
        {
            "metric": {
                "model": {
                    "scores": {...}
                },
                "model": {
                    "scores": {...}
                },
                ...
            }
        },
        ...
    ]
    """

    # loop over metrics
    all_image_evals = []
    for metric in [
        "bleu-1",
        "bleu-2",
        "bleu-3",
        "bleu-4",
        "meteor",
        "rouge",
        "cider",
        # "spice",
        "bertscore",
        "clipscore",
        "clipscore_ref",
    ]:
        start_time = time.time()
        print(f"Evaluating {metric}...")
        current_eval = []

        # check if the data has already been evaluated
        already_evaluated = check_if_evaluated(data, metric)
        if already_evaluated is None:
            # loop over models
            for model in candidates.keys():
                # finally, compute the scores for the current model and metric
                print(f"---for {model}...", end="", flush=True)
                if metric == "bleu-1":
                    order = 1
                    scores = execute_bleu(candidates[model], references, order)
                elif metric == "bleu-2":
                    order = 2
                    scores = execute_bleu(candidates[model], references, order)
                elif metric == "bleu-3":
                    order = 3
                    scores = execute_bleu(candidates[model], references, order)
                elif metric == "bleu-4":
                    order = 4
                    scores = execute_bleu(candidates[model], references, order)
                elif metric == "meteor":
                    scores = execute_meteor(candidates[model], references)
                elif metric == "rouge":
                    scores = execute_rouge(candidates[model], references)
                elif metric == "bertscore":
                    scores = execute_bertscore(candidates[model], references)
                elif metric == "cider":
                    average, scores = execute_cider(candidates[model], references)
                    scores = [{"score": x} for x in scores]
                elif metric == "spice":
                    average, scores = execute_spice(candidates[model], references)
                    print(f"SPICE: {average}")
                    print(f"SPICE scores: {scores}")
                elif metric == "clipscore":
                    scores = execute_clipscore(candidates[model], image_files)
                elif metric == "clipscore_ref":
                    scores = execute_clipscore_ref(
                        candidates[model], references, image_files
                    )

                # add to merged_evals
                if len(current_eval) == 0:
                    for index, score in enumerate(scores):
                        current_eval.append({metric: {model: score}})
                else:
                    for index, score in enumerate(scores):
                        current_eval[index][metric][model] = score

                print(f"complete.")
        else:
            print(f"Data already evaluated for {metric}. Skipping.")
            current_eval = [{metric: x} for x in already_evaluated]

        # merge results with all_image_evals
        if len(all_image_evals) == 0:
            all_image_evals = copy.deepcopy(current_eval)
        else:
            # merge the current eval dictionary with the all_image_evals dictionary
            for index, eval in enumerate(current_eval):
                all_image_evals[index] = {**all_image_evals[index], **eval}

        print(f"{metric} complete. Took {time.time() - start_time:.4f} seconds.\n")

    # attach all_image_evals to the data object
    for index, image in enumerate(data):
        # check if evaluation exists and save score
        if "evaluation" not in image:
            image["evaluation"] = {}
        image["evaluation"] = copy.deepcopy(all_image_evals[index])

    # export results
    # check if output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # create an output path by using the file name from the input path and output directory
    output_path = os.path.join(
        args.output_dir,
        f"evaluation_results_{len(data)}-images_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.json",
    )

    # save the data to the output path
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
