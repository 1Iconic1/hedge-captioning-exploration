"""
Generate input data for the CHI 2026 paper, using annotated high- and low-quality data from VizWiz and selected marketing images.

Example usage:
python generate_input_data.py \
    --low-quality-data-path ./input-data/low-quality-images_08-24-25.csv \
    --high-quality-data-path ./input-data/high-quality-images_08-24-25.csv \
    --marketing-data-path ./input-data/matched-product-images_08-24-25.csv \
    --output-dir ./input-data/ \
    --should-randomize True
"""

from datetime import datetime
from enum import Enum
import os
import argparse
import pandas as pd


def fill_blank_values(data: pd.DataFrame):
    """Fill blank values in a pandas dataframe.

    Args:
        data (pandas.DataFrame): Data to fill.

    Returns:
        None: In-place modification of the dataframe.
    """
    for col in data.columns:
        # get the type of the column
        col_type = data[col].dtype
        if col_type == "object":
            data.fillna({col: ""}, inplace=True)
        elif col_type == "float64":
            data.fillna({col: 0}, inplace=True)
        elif col_type == "Int8":
            data.fillna({col: 0}, inplace=True)
        elif col_type == "boolean" or col_type == "bool":
            data.fillna({col: False}, inplace=True)
        else:
            data.fillna("", inplace=True)


# load the data
def load_low_quality_data(low_quality_data_path: str):
    """Load low-quality data from a CSV file.

    Args:
        low_quality_data_path (str): Path to the low-quality data CSV file.

    Returns:
        pandas.DataFrame: Low-quality data.
    """
    # load the data
    low_quality_df = pd.read_csv(low_quality_data_path)

    # column formatting
    low_quality_df["image_preview"] = ""
    low_quality_df["type"] = "low-quality"
    low_quality_df = low_quality_df.rename(columns={"vizwiz_url": "image_url"})
    fill_blank_values(low_quality_df)

    # filter data
    original_size = len(low_quality_df)
    low_quality_df = low_quality_df[
        (low_quality_df["unable_to_verify"] == "") & (low_quality_df["exclude?"] == "")
    ]
    print(
        f"BLV Low-Quality Images | Original Size {original_size} | Filtered Size {len(low_quality_df)}"
    )

    # return the data
    return low_quality_df


def load_high_quality_data(high_quality_data_path: str):
    """Load high-quality data from a CSV file.

    Args:
        high_quality_data_path (str): Path to the high-quality data CSV file.

    Returns:
        pandas.DataFrame: High-quality data.
    """
    # load the data
    high_quality_df = pd.read_csv(high_quality_data_path)

    # column formatting
    high_quality_df["image_preview"] = ""
    high_quality_df["type"] = "high-quality"
    high_quality_df = high_quality_df.rename(
        columns={"vizwiz_url": "image_url", "image_id": "id"}
    )
    fill_blank_values(high_quality_df)

    # filter data
    original_size = len(high_quality_df)
    high_quality_df = high_quality_df[(high_quality_df["exclude?"] == "")]
    print(
        f"BLV High-Quality Images | Original Size {original_size} | Filtered Size {len(high_quality_df)}"
    )

    # return the data
    return high_quality_df


def load_marketing_data(marketing_data_path: str):
    """Load marketing data from a CSV file.

    Args:
        marketing_data_path (str): Path to the marketing data CSV file.

    Returns:
        pandas.DataFrame: Marketing data.
    """
    # load the data
    marketing_df = pd.read_csv(marketing_data_path)

    # column formatting
    marketing_df["image_preview"] = ""
    marketing_df["type"] = "matched-image"
    del marketing_df["google_drive_file"]
    fill_blank_values(marketing_df)

    # filter data
    original_size = len(marketing_df)
    marketing_df = marketing_df[(marketing_df["was excluded?"] == "")]
    print(
        f"Product Marketing Images | Original Size {original_size} | Filtered Size {len(marketing_df)}"
    )

    # return the data
    return marketing_df


def combine_data(
    low_quality_data: pd.DataFrame,
    high_quality_data: pd.DataFrame,
    marketing_data: pd.DataFrame,
    should_randomize: bool = False,
):
    """Combines data into a single list of dictionaries.

    Args:
        low_quality_data (pd.DataFrame): Low-quality data.
        high_quality_data (pd.DataFrame): High-quality data.
        marketing_data (pd.DataFrame): Marketing data.
        should_randomize (bool, optional): Whether to randomize the order of the data. Defaults to False.

    Returns:
        list[dict]: Combined data.
    """
    # create a pandas dataframe with all dataframes combined
    output_df = pd.concat(
        [
            low_quality_data,
            high_quality_data,
            marketing_data,
        ]
    )

    # replace missing values with empty strings
    fill_blank_values(output_df)

    # make annotation columns lowercase and trimmed
    for col in ["object", "product", "brand", "variety"]:
        output_df[col] = output_df[col].str.lower().str.strip()

    # clean-up columns
    output_df["annotator"] = (
        output_df["annotator"].fillna("") + output_df["new annotator"]
    ).fillna("")
    output_df["annotation notes"] = (
        output_df["annotation notes"].fillna("")
        + output_df["new annotation notes"].fillna("")
    ).fillna("")

    # don't replace empty with False since since some of the data doesn't have annotations
    output_df["curved label"] = (
        output_df["curved label"].astype(str).replace({"x": "True"})
    )
    output_df["text panel"] = output_df["text panel"].astype(str).replace({"x": "True"})

    # replace empty with False for quality issue columns
    for col in [
        "unrecognizable",
        "framing",
        "blur",
        "obstruction",
        "rotation",
        "too dark",
        "too bright",
        "other",
    ]:
        output_df[col] = output_df[col].astype(str).replace({"": "False"})

    # there will always be text, given how we select products
    output_df["text_detected"] = (
        output_df["text_detected"].astype(str).replace({"": "True"})
    )

    # remove unneeded columns
    for col in [
        "gpt4o_caption",
        "gpt4o_code",
        "llama_caption",
        "llama_code",
        "molmo_caption",
        "molmo_code",
        "new annotator",
        "new annotation notes",
    ]:
        del output_df[col]

    # reorder columns
    column_info = [
        ("id", "string"),
        ("orig_id", "string"),
        ("file_name", "string"),
        ("image_url", "string"),
        ("image_preview", "string"),
        ("type", "string"),
        ("human_captions", "string"),
        ("expert_caption", "string"),
        ("orig annotator", "string"),
        ("orig annotation notes", "string"),
        ("unable_to_verify", "string"),
        ("double code notes", "string"),
        ("double verified", "string"),
        ("annotator", "object"),
        ("annotation notes", "object"),
        # "exclude?", "string"),
        # "was excluded?", "string"),
        ("object", "string"),
        ("product", "string"),
        ("brand", "string"),
        ("variety", "string"),
        ("double annotator", "string"),
        ("double annotation", "string"),
        ("text_detected", "string"),
        ("curved label", "string"),
        ("text panel", "string"),
        ("unrecognizable", "string"),
        ("framing", "string"),
        ("blur", "string"),
        ("obstruction", "string"),
        ("rotation", "string"),
        ("too dark", "string"),
        ("too bright", "string"),
        ("other", "string"),
        # "AMP_rotation", "string"),
        # "XT_rotation", "string"),
        ("unrecognizable_orig", "float64"),
        ("framing_orig", "float64"),
        ("blur_orig", "float64"),
        ("obstruction_orig", "float64"),
        ("rotation_orig", "float64"),
        ("too_dark_orig", "float64"),
        ("too_bright_orig", "float64"),
        ("other_orig", "float64"),
        ("no_issue_orig", "float64"),
    ]
    output_df = output_df[[col for col, _ in column_info]]

    # set datatypes for columns
    for col, dtype in column_info:
        output_df[col] = output_df[col].astype(dtype)

    # randomize the data
    if should_randomize:
        output_df = output_df.sample(frac=1).reset_index(drop=True)

    print(f"Combined Data | Size {len(output_df)}")
    return output_df


class DataFormat(Enum):
    JSON = "json"
    CSV = "csv"


def save_data(
    data: pd.DataFrame,
    output_dir: str,
    output_filename: str,
    format: DataFormat = DataFormat.JSON,
):
    """Save the combined data to a file.

    Args:
        data (pd.DataFrame): DataFrame to save.
        output_path (str): Path to save the data.
        format (DataFormat, optional): Format to save the data. Defaults to DataFormat.JSON.
    """
    # check that the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{os.path.join(output_dir, output_filename)}.{format.value}"

    # save the data
    if format == DataFormat.JSON:
        with open(output_path, "w") as f:
            data.to_json(f, orient="records", indent=4)
    elif format == DataFormat.CSV:
        data.to_csv(output_path, index=False)

    # print the number of images in the data
    print(f"Saved data to {output_path}. Includes {len(data)} images.")


def parse_args():
    """Parse input arguments.

    Returns:
        argparse.Namespace: Arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-quality-data-path", type=str, required=True)
    parser.add_argument("--high-quality-data-path", type=str, required=True)
    parser.add_argument("--marketing-data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--should-randomize", type=bool, default=False)
    return parser.parse_args()


def main():
    # get arguments
    args = parse_args()

    # load the data
    low_quality_images_df = load_low_quality_data(args.low_quality_data_path)
    high_quality_images_df = load_high_quality_data(args.high_quality_data_path)
    marketing_images_df = load_marketing_data(args.marketing_data_path)

    # combine the data
    combined_images_df = combine_data(
        low_quality_images_df,
        high_quality_images_df,
        marketing_images_df,
        should_randomize=args.should_randomize,
    )

    # save the data as a JSON file and a CSV file
    save_data(
        combined_images_df,
        args.output_dir,
        f"combined-image-input_{len(combined_images_df)}-images_{datetime.now().strftime('%Y-%m-%d')}",
        format=DataFormat.JSON,
    )

    save_data(
        combined_images_df,
        args.output_dir,
        f"combined-image-input_{len(combined_images_df)}-images_{datetime.now().strftime('%Y-%m-%d')}",
        format=DataFormat.CSV,
    )


if __name__ == "__main__":
    main()
