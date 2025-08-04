# Analyzing Issues With Image Captioning for BLV People

Studying the effective of image quality issues for blind and low-vision (BLV) people when using Vision Language Models (VLMs) for image captioning.

## Prerequisites

### Software requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- [Hugging Face](https://huggingface.co/) account for models. You may need to request access to:
  - [allenai/Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924)
  - [allenai/Molmo-72B-0924](https://huggingface.co/allenai/Molmo-72B-0924)
  - [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
  - [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)
- [OpenAI API](https://openai.com/index/openai-api/) Key
- [Gemini API](https://ai.google.dev/) Key

### Computational hardware requirements

- Multiple modules use PyTorch or Hugging Face Transformers and will be significantly more performant with a CUDA GPU or Apple Silicon chip. All code is written to fall back to CPU when neither is available, but expect very slow inference without a GPU.
- Code for running Llama and Molmo models requires ~35-40 GB of VRAM.

## Setup

### Accessing models

1. Follow the instruction on [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) to login with your Hugging Face account.

1. Create an `.env` file with the following:

   ```bash
   OPENAI_API_KEY=<OPEN_API_KEY_HERE>
   GEMINI_API_KEY=<GEMINI_API_KEY_HERE>

   ```

### Getting Started

1. Run `uv sync` to setup your virtual environment and install packages.
1. Run `pre-commit install` to setup commit hooks.
1. Download data into `./data/`. Ask Kapil what data is needed.
1. Activate your virtual env using `source .venv/bin/activate`.
1. Run `python -m spacy download en_core_web_sm` to download SpaCy model.
   - All other models will be downloaded when running code that requires them.

## Development

- Add new packages using `uv add <package>`. If packages can only run on some platforms, make sure to specify them in `pyproject.toml`; see [uv's PyTorch installation](https://docs.astral.sh/uv/guides/integration/pytorch/), for example.
- When commiting code, [`ruff`](https://docs.astral.sh/ruff/) will automatically lint and format your code. If linting or formatting errors are found, they will be fixed automatically, but you will need to re-commit the changes after verifying them. See [example of process here](https://medium.com/@kutayeroglu/automate-python-formatting-with-ruff-and-pre-commit-b6cd904b727e) for an example.

## Repository walkthrough -- NEEDS UPDATING

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
