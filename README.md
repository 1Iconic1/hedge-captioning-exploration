# Image Captioning Experiments for BLV Poople

Studying the effective of image quality issues for blind and low-vision (BLV) people when using Vision Language Models (VLMs) for image captioning.

## Prerequisites

- Python 3.12
- conda

## Installation

1. Run `conda env create -f environment.yml` to install dependencies.
2. Download data into `./data/`. Ask Kapil what data is needed.

## Development

Add new packages using `conda install <package>`. Make sure to run `conda env export | grep -v "^prefix: " > environment.yml` to save the updated environment to version control.

## Repository walkthrough

The repository contains multiple experiments across jupyter notebooks and scripts designed to run on a [slurm-managed server](https://slurm.schedmd.com/documentation.html). Code is designed to use GPUs that are available, and has been tested with CURA and Apple Silicon. Models are sourced from Huggingface.

```bash
.
├── README.md
├── data
│   ├── caption-dataset
│   ├── clean-images
│   ├── image-quality-assessment
│   ├── multi-generation-experiment
│   ├── obscured-images
│   ├── privacy-images
│   ├── rotation-experiment-images
│   └── labeled-data # output
├── environment.yml
├── llama-experiment-notebooks # experiments run with Meta's Llama 3.2 vision instruct
├── molmo-experiment-notebooks # experiments run with AllenAI's Molmo model
├── scripts # scripts to run on server
└── slurm-config # configuration to run scripts with slurm workload manager
```
