# illama
**illama** is a lightweight, fast inference server for Llama and ExLlamav2 based large language models (LLMs).

## Features
- **Continuous batching** - Handles multiple requests simultaneously.
- **Open-AI compatible server** - Use official OpenAI API clients
- **Quantization Support** - Load any quantized ExLlamaV2 compatible models (GPTQ, EXL2, or SafeTensors).
- **GPU Focused** - Distribute model across any number of local GPUs.
- Uses [FlashAttention 2](https://github.com/Dao-AILab/flash-attention) with Paged Attention by default

## Installation

To get started, clone the repo.

```bash
git clone https://github.com/nickpotafiy/illama.git
cd illama
```

### With Conda

Optionally, create a new conda environment.

```bash
conda create -n illama python=3.10
conda activate illama
```

### Requirements

Install the required packages.

```bash
pip3 install -r requirements.txt
```

This will install [ExLlamaV2](https://github.com/turboderp/exllamav2) and the required libraries.

## Running the Server

To start illama run this command:

```bash
python server.py --model-path "<path>" --batch-size 10 --host "0.0.0.0" --port 5000 --verbose
```
