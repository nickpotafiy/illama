from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Tokenizer
)
from illama import ILlamaServer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, help="The path to a GPTQ, EXL2, or SafeTensor model")
parser.add_argument("-b", "--batch-size", type=int, default=2, help="The batch size (default: 2)")
parser.add_argument("-H", "--host", type=str, default="127.0.0.1", help="The host address (default: 127.0.0.1)")
parser.add_argument("-p", "--port", type=int, default=5000, help="The port number (default: 5000)")
parser.add_argument("-v", "--verbose", action="store_true", help="Print extra details (default: False)")
parser.add_argument("-c", "--cache-quantization", type=int, choices=[16, 8, 4], default=16, help="The cache quantization options: 16, 8, or 4 bit (default: 16)")

args = parser.parse_args()

model_path = args.model
batch_size = args.batch_size
host = args.host
port = args.port
verbose = args.verbose
cache_quant = args.cache_quantization

config = ExLlamaV2Config(model_dir=model_path)
config.prepare()
tokenizer = ExLlamaV2Tokenizer(config)
model = ExLlamaV2(config)

print("Cache quant:", cache_quant)

if cache_quant == 16:
    cache = ExLlamaV2Cache(model, batch_size, lazy=True)
elif cache_quant == 8:
    cache = ExLlamaV2Cache_8bit(model, batch_size, lazy=True)
elif cache_quant == 4:
    cache = ExLlamaV2Cache_Q4(model, batch_size, lazy=True)
else:
    print("Unsupported cache quant", cache_quant)
    exit(0)

model.load_autosplit(cache)

server = ILlamaServer(host, port, model, tokenizer, cache, verbose)
server.serve()