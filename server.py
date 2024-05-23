from illama import ILlamaServer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-path", type=str, required=True, help="The path to a GPTQ, EXL2, or SafeTensor model")
parser.add_argument("-b", "--batch-size", type=int, default=2, help="The batch size (default: 2)")
parser.add_argument("-H", "--host", type=str, default="127.0.0.1", help="The host address (default: 127.0.0.1)")
parser.add_argument("-p", "--port", type=int, default=5000, help="The port number (default: 5000)")
parser.add_argument("-v", "--verbose", action="store_true", help="Print extra details (default: False)")
parser.add_argument("-c", "--cache-quantization", type=int, choices=[16, 8, 4], default=16, help="The cache quantization options: 16, 8, or 4 bit (default: 16)")

args = parser.parse_args()

host = args.host
port = args.port
model_path = args.model_path
batch_size = args.batch_size
cache_quant = args.cache_quantization
verbose = args.verbose

server = ILlamaServer(host, port, model_path, batch_size, cache_quant, verbose)
server.serve()