# Analyzing Issues With Image Captioning for BLV People

Studying the effective of image quality issues for blind and low-vision (BLV) people when using Vision Language Models (VLMs) for image captioning.

## Prerequisites

- Python 3.11
- Pipenv

## Setup

- You will need an OpenAI API key
- Multiple modules use PyTorch and will be significantly more performant with a CUDA GPU or Apple Silicon chip. All code is written to fall back to CPU when neither is available.
- Code for running Llama and Molmo models requires ~35-40 GB of memory.

## Installation

1. Run `pipenv install` to install dependencies.
2. Download data into `./data/`. Ask Kapil what data is needed.
3. Create a `.env` file and add: `OPENAI_API_KEY=<YOUR KEY HERE>`
4. Run `python -m spacy download en_core_web_sm` to download SpaCy model.

## Development

Add new packages using `pipenv install <package>`.

## Repository walkthrough

The repository contains multiple experiments across jupyter notebooks and scripts designed to run on a [slurm-managed server](https://slurm.schedmd.com/documentation.html). Code is designed to use GPUs that are available, and has been tested with CUDA and Apple Silicon. Models are sourced from Huggingface.

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
