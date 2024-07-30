# illama
**illama** is a lightweight, fast inference server for Llama and ExLlamav2 based large language models (LLMs).

## Features
- **Continuous batching** - Handles multiple requests simultaneously.
- **Open-AI compatible server** - Use official OpenAI API clients
- **Quantization Support** - Load any quantized ExLlamaV2 compatible models (GPTQ, EXL2, or SafeTensors).
- **GPU Focused** - Distribute model across any number of local GPUs.
- Uses [FlashAttention 2](https://github.com/Dao-AILab/flash-attention) with Paged Attention by default

## Getting Started

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

## Install PyTorch

Install [Nvidia Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) and [PyTorch](https://pytorch.org/get-started/locally/). Ideally, both versions should match to minimize incompatibilities. PyTorch CUDA `12.1` is recommended with Nvidia CUDA Toolkit 12.1+.

### Install Torch w/ Pip

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Install Torch w/ Conda

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Check Torch CUDA version with: `python -c "import torch; print(torch.version.cuda)"`

### Install Illama

First, install setup libraries:

```bash
pip install packaging ninja
```

Then, install the main package:

```bash
pip install .
```

If installation fails, you may need to `set MAX_JOBS=4` or `export MAX_JOBS=4` (or lower) depending on system memory. This is a known `flash-attn` problem.

## Running the Server

To start illama server, run this command:

```bash
python server.py --model-path "<path>" --batch-size 10 --host "0.0.0.0" --port 5000 --verbose
```

Run `python server.py --help` to get a list of all available options.

## Troubleshooting

If you get an error saying `OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root`, that typically means PyTorch was not installed correctly. You can verify PyTorch installation by activating your environment and executing `python`:

```python
import torch
torch.version.cuda
```
If you don't get your PyTorch CUDA version, then it was not installed correctly. You may have installed PyTorch without CUDA (like a Preview build).