# Inference Llama
**iLlama** is a fast inference server for running Llama and ExLlamav2 based large language models (LLMs).

## Features
- Dynamic batching - Handles multiple completion requests simultaneously.
- Open-AI compatible server - Use official API clients to connect to the server.
- Load any ExLlamaV2 compatible models (GPTQ, EXL2, or SafeTensors).
- Quantization support with GPTQ and EXL2.
- iLlama uses ExLlamaV2's blazing fast library internally.
- Distribute model across any number of local GPUs.