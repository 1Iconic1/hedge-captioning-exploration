# Libraries
import csv


def read_caption(caption_file):
    """
    Read csv file that stores groundtruth captions and generated captions from different models.

    Inputs:
    - caption_file: csv file that stores caption

    Output:
    - list(dictionary): dictionary key: [human_caption_0, human_caption_1, human_caption_2, human_caption_3, human_caption_4, molmo_model_caption, gpt4_model_caption, llama_model_caption]
    """
    caption_dataset = {
        "human_caption_0": [], 
        "human_caption_1": [], 
        "human_caption_2": [], 
        "human_caption_3": [], 
        "human_caption_4": [], 
        "molmo_model_caption": [], 
        "gpt4_model_caption": [], 
        "llama_model_caption": []
    }
    with open(caption_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            caption_dataset["human_caption_0"] = row[0]
            caption_dataset["human_caption_1"] = row[1]
            caption_dataset["human_caption_2"] = row[2]
            caption_dataset["human_caption_3"] = row[3]
            caption_dataset["human_caption_4"] = row[4]
            caption_dataset["molmo_model_caption"] = row[5]
            caption_dataset["gpt4_model_caption"] = row[6]
            caption_dataset["llama_model_caption"] = row[7]

    return caption_dataset


def generate_atomic_statement(org_caption, ground_truth=False):
    """
    Generates atomic statements from original sentence.

    Inputs:
    - org_caption: original caption
    - ground_truth: True(T, the human generated caption), False(g, model generated caption)

    Output:
    - list(str): statements from 
    """
    output = list()

    return output

def save_atomic_statements(caption_dataset_filename, image_quality_dataset_filename):
    """
    Save genearated atomic statements

    Inputs:
    - caption_dataset_filename (str): path to caption dataset.
    - image_quality_dataset_filename (str): path to image quality dataset.

    Output:
    - (pd.DataFrame): dataframe containing image annotations and image quality.
    """

def main():
    print("Read caption csv file...")
    read_caption("dummy_caption.csv")

    print("Generating atomic statements...")
    #TODO 

    # save atomic statements
     #TODO


if __name__ == "__main__":
    main()
