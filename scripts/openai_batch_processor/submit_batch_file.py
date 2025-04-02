"""
This script is used to submit a batch file to the OpenAI Batch API.

Example usage:
python submit_batch_file.py \
    --input-file ./batch_input/gpt-4o-2024-08-06-captioning-batch-input_2025-03-29_04:10:36.jsonl

"""

import argparse
import os
import json

from pathlib import Path
from datetime import datetime

from openai import OpenAI

client = OpenAI()


def upload_batch_input_file(batch_input_file_path):
    """
    Upload a batch input file to the OpenAI Batch API.

    Args:
        batch_input_file_path (str): The path to the batch input file.

    Returns:
        dict: The batch input file.
    """
    batch_input_file = client.files.create(
        file=open(batch_input_file_path, "rb"), purpose="batch"
    )

    return batch_input_file


def create_batch_job(batch_input_file, batch_description):
    """
    Create a batch job for the OpenAI Batch API.

    Args:
        batch_input_file (dict): The batch input file.
        batch_description (str): The description of the batch job.
    Returns:
        dict: The batch job.
    """
    # submit the batch file to the OpenAI Batch API
    return client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"description": batch_description},
    )


def save_submitted_batch_job(batch_job, output_batch_job_path):
    """
    Save the submitted batch job to a file.

    Args:
        batch_job (dict): The batch job.
        output_batch_job_path (str): The path to save the batch job.

    Returns:
        None
    """
    with open(output_batch_job_path, "w") as f:
        json.dump(batch_job, f, indent=4)


def main():
    """
    Main function to submit a batch file to the OpenAI Batch API.
    """
    # get the batch input file path
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="")
    args = parser.parse_args()

    # get filepath and filename for input
    batch_file_name = Path(args.input_file).stem
    input_batch_file_path = args.input_file

    # upload the batch input file
    batch_input_file = upload_batch_input_file(input_batch_file_path)

    # schedule the batch job and save the batch job to a file
    batch_description = f"Captioning job for {batch_file_name}"
    batch_job = create_batch_job(batch_input_file, batch_description)

    os.makedirs("./batch_jobs", exist_ok=True)
    output_batch_job_path = (
        args.output_file
        if args.output_file != ""
        else f"./batch_jobs/{batch_file_name}_submitted-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json"
    )
    batch_job_dict = batch_job.to_dict()
    save_submitted_batch_job(batch_job_dict, output_batch_job_path)

    # print the batch job info
    print(
        f"Batch job submitted. File saved to {output_batch_job_path}. \n Batch Job: {json.dumps(batch_job_dict, indent=4)}"
    )


if __name__ == "__main__":
    main()
