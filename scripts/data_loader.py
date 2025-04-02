import json
import pandas as pd


def generate_target_dataset(caption_dataset_filename, image_quality_dataset_filename):
    """
    Generates a target dataset for captioning based on VizWiz's image captioning dataset and image quality assessment dataset.

    Inputs:
    - caption_dataset_filename (str): path to caption dataset.
    - image_quality_dataset_filename (str): path to image quality dataset.

    Output:
    - (list of dict): image annotations and image quality.
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

    # remove duplicate id column
    image_captioning_input.drop(["id"], axis=1, inplace=True)

    # reorder columns
    image_captioning_input = image_captioning_input[
        [
            "image_id",
            "file_name",
            "vizwiz_url",
            "text_detected",
            "unrecognizable",
            "framing",
            "blur",
            "obstruction",
            "rotation",
            "too dark",
            "too bright",
            "other",
            "no issue",
            "caption",
            "is_precanned",
            "is_rejected",
        ]
    ]

    # convert image_captioning_input to a list of dictionaries
    image_captioning_input = image_captioning_input.to_dict(orient="records")

    # expand captions, is_precanned, and is_rejected into individual columns
    for index, row in enumerate(image_captioning_input):
        curr_captions = row["caption"]
        curr_precanned = row["is_precanned"]
        curr_rejected = row["is_rejected"]

        # expand captions
        human_captions = []
        for caption_index in range(0, len(curr_captions)):
            curr_human_caption = {
                "caption": curr_captions[caption_index],
                "is_precanned": curr_precanned[caption_index],
                "is_rejected": curr_rejected[caption_index],
            }
            human_captions.append(curr_human_caption)

        image_captioning_input[index]["human_captions"] = human_captions

        # remove old rows
        del image_captioning_input[index]["caption"]
        del image_captioning_input[index]["is_precanned"]
        del image_captioning_input[index]["is_rejected"]

    return image_captioning_input


def filter_dataset(
    image_captioning_input,
    threshold=4,
    issues_to_filter=[
        "blur",
        "framing",
        "obstruction",
        "rotation",
        "too dark",
        "too bright",
    ],
):
    """
    Filters the dataset based on the quality columns.

    Inputs:
    - image_captioning_input (pd.DataFrame): dataframe containing image annotations and image quality.
    - quality_columns (list): list of quality columns to filter on.
    - threshold (int): threshold for the quality columns. Must be between 0 to 5. Filtered value is >= threshold.
    - issues_to_filter (list): list of quality columns to filter on.

    Output:
    - (list of dict): filtered image annotations and image quality.
    """
    # select images where 3 or more people could provide a caption (unrecognizable < 3)
    target_subset_df = image_captioning_input[
        image_captioning_input["unrecognizable"] < 3
    ]

    # filter on image quality where threshold or more people said the image quality issue was present
    # filter is an OR filter on all issues_to_filter
    target_subset_df = target_subset_df[
        (target_subset_df[issues_to_filter] >= threshold).any(axis=1)
    ]

    # return filtered dataset
    return target_subset_df.to_dict(orient="records")


if __name__ == "__main__":
    image_captioning_input = generate_target_dataset(
        "../data/caption-dataset/annotations/train.json",
        "../data/image-quality-assessment/annotations/train.json",
    )
    filtered_dataset = filter_dataset(pd.DataFrame.from_dict(image_captioning_input))
    print(json.dumps(filtered_dataset[0], indent=4))
