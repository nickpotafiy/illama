from exllamav2.cache import ExLlamaV2Cache
from exllamav2.config import ExLlamaV2Config
from exllamav2.model import ExLlamaV2
from exllamav2.tokenizer.tokenizer import ExLlamaV2Tokenizer
from illama import ILlamaServer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="The path to a GPTQ, EXL2, or SafeTensor model")
parser.add_argument("-b", "--batch-size", type=int, default=2, help="The batch size (default: 2)")
parser.add_argument("-H", "--host", type=str, default="127.0.0.1", help="The host address (default: 127.0.0.1)")
parser.add_argument("-p", "--port", type=int, default=5000, help="The port number (default: 5000)")
parser.add_argument("-v", "--verbose", action="store_true", help="Print extra details (default: False)")

args = parser.parse_args()

model_path = args.model
batch_size = args.batch_size
host = args.host
port = args.port
verbose = args.verbose

if not model_path:
    print("--model is required")
    exit(0)

config = ExLlamaV2Config(model_dir=model_path)
config.prepare()
tokenizer = ExLlamaV2Tokenizer(config)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, batch_size, lazy=True)
model.load_autosplit(cache)

server = ILlamaServer(host, port, model, tokenizer, cache, verbose)
server.serve()