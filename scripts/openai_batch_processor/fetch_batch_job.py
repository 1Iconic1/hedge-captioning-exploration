"""
This script is used to fetch a batch job from the OpenAI Batch API.
It first checks if the batch job has completed. If it has, it fetches the batch job and saves the raw batch job's output to a file.

Example usage:
# with batch job file
python fetch_batch_job.py \
    --batch-job-file ./batch_jobs/gpt-4o-2024-08-06-captioning-batch-input_start-0_end-1000_2025-03-29_05:13:54_submitted-2025-03-29_05:14:08.json \
    --output-file-name gpt-4o-2024-08-06-captioning-batch-input_start-0_end-1000_2025-03-29_05:13:54_completed-2025-03-29_05:14:08.jsonl

# with batch job id
python fetch_batch_job.py \
    --batch-job-id <batch-job-id>
"""

import argparse
import os
import json
from openai import OpenAI

client = OpenAI()


def check_batch_job_status(batch_job_id):
    """
    Check the status of a batch job.

    Args:
        batch_job_id (str): The id of the batch job.

    Returns:
        dict: The batch job.
    """
    batch_job = client.batches.retrieve(batch_job_id)
    return batch_job.to_dict()


def fetch_batch_job_output(output_file_id, output_file_name):
    """
    Fetch the output of a batch job.

    Args:
        output_file_id (str): The id of the output file.
        output_file_name (str): The name of the output file.

    Returns:
        str: The content of the output file.
        path: The path to the output file.
    """
    # TODO: include option to fetch batch data if successful.
    file_response = client.files.content(output_file_id)
    with open(output_file_name, "w") as f:
        f.write(file_response.text)

    return file_response.text, output_file_name


def main():
    """
    Main function to fetch a batch job from the OpenAI Batch API.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-job-file", type=str, default="")
    parser.add_argument("--batch-job-id", type=str, default="")
    parser.add_argument("--output-file-name", type=str, default="")
    args = parser.parse_args()

    if args.batch_job_file != "":
        with open(args.batch_job_file, "r") as f:
            batch_job = json.load(f)
        batch_job_id = batch_job["id"]
    elif args.batch_job_id != "":
        batch_job_id = args.batch_job_id
    else:
        raise ValueError("Either batch-job-file or batch-job-id must be provided.")

    # fetch the batch job
    batch_job = check_batch_job_status(batch_job_id)

    # if the batch job is successful, fetch the output
    if batch_job["status"] == "completed":
        print(f"Batch job {batch_job_id} is completed. Fetching output...")
        output_file_id = batch_job["output_file_id"]

        # fetch and save the output
        output_dir = "./batch_output"
        output_file_name = (
            args.output_file_name
            if args.output_file_name != ""
            else f"{batch_job_id}.jsonl"
        )
        output_file_path = os.path.join(output_dir, output_file_name)

        os.makedirs(output_dir, exist_ok=True)
        fetch_batch_job_output(output_file_id, output_file_path)
        print(f"Batch job {batch_job_id}'s output is saved to {output_file_path}.")
    else:
        print(f"Batch job {batch_job_id} is not completed.")
        print(json.dumps(batch_job, indent=4))


if __name__ == "__main__":
    main()
