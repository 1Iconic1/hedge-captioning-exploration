"""
Runs all evaluation metrics on the captions and outputs a json file with the results..\
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input JSON file path.")
    parser.add_argument("--output", type=str, help="Output JSON file path.")
    args = parser.parse_args()

    # load the data from the files
    data = load_data(args.input)

    # check if output directory exists
    os.makedirs(args.output, exist_ok=True)

    # create an output path by using the file name from the input path and output directory
