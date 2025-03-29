"""
Convert a JSON file with all captioning data into a human-readable CSV.
"""

import os
import json
import pandas as pd
import argparse


def load_data(file_path):
    """Load the data from the given file path."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def convert_data_to_csv(data, output_path):
    """
    Convert the data to a CSV.

    Args:
        data (list): The data to convert.
        output_path (str): The path to save the CSV file.

    Returns:
        pd.DataFrame: The converted data.
    """
    models = []
    for item in data:
        # split human captions into separate columns
        # can ignore precanned and rejected columns for human-readable csv
        for index, caption in enumerate(item["human_captions"]):
            item[f"human_caption_{index}"] = caption["caption"]

        # split model captions into separate columns
        for index, model_caption in enumerate(item["model_captions"]):
            item[f"{model_caption['model_name']}_caption"] = model_caption["caption"]
            models.append(model_caption["model_name"])

        # TOOD: when performance metrics are added, extract them into separate columns

        # add columns for notes
        item["general_notes"] = ""
        for model in models:
            item[f"{model}_notes"] = ""

        # remove the original columns
        item.pop("human_captions")
        item.pop("model_captions")

    # convert to csv
    df = pd.DataFrame(data)

    # save to csv
    df.to_csv(output_path, index=False)

    return df


def main():
    # get files from user using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input JSON file path.")
    parser.add_argument("--output", type=str, help="Output CSV directory path.")
    args = parser.parse_args()

    # load the data from the files
    data = load_data(args.input)

    # check if output directory exists
    os.makedirs(args.output, exist_ok=True)

    # create an output path by using the file name from the input path and output directory
    output_path = os.path.join(
        args.output, os.path.basename(args.input).replace(".json", ".csv")
    )

    # convert the data to a CSV
    convert_data_to_csv(data, output_path)
    print(f"CSV saved to {output_path}")


if __name__ == "__main__":
    """
    Example usage:
    python convert_json_to_csv.py \
        --input ../../data/study-2-output/labeled-data/combined-caption-output/combined-caption-output_50-images2025-03-28_17:36:12.json \
        --output ../../data/study-2-output/labeled-data/combined-caption-output/
    """
    main()
