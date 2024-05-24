# iLlama
**iLlama** is a fast inference server for running Llama and ExLlamav2 based large language models (LLMs). The main feature of iLlama (and the reason it was made) is parallel processing of requests to output tokens for multiple users at once.

## Features
- Dynamic batching - Handles multiple completion requests simultaneously.
- Open-AI compatible server - Use official API clients to connect to the server.
- Load any ExLlamaV2 compatible models (GPTQ, EXL2, or SafeTensors).
- Quantization support with GPTQ and EXL2.
- iLlama uses ExLlamaV2's blazing fast library internally.
- Nvidia GPU focused - Distribute model across any number of local GPUs.

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

To start the server run this command and specify the path to your model.

```bash
python server.py --model-path "<path>"
```
