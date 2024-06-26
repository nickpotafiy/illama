from illama import IllamaServer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-path", type=str, required=True, help="The path to a GPTQ, EXL2, or SafeTensor model")
parser.add_argument("-b", "--batch-size", type=int, default=2, help="The batch size (default: 2)")
parser.add_argument("-H", "--host", type=str, default="127.0.0.1", help="The host address (default: 127.0.0.1)")
parser.add_argument("-p", "--port", type=int, default=5000, help="The port number (default: 5000)")
parser.add_argument("-c", "--max-chunk-size", type=int, default=256, help="Maximum number of tokens to process in parallel during prefill (prompt ingestion). Should not exceed the model's max_input_len but can be lowered to trade off prompt speed for a shorter interruption to ongoing jobs when a new job is started. Defaults to 256.")
parser.add_argument("-v", "--verbose", action="store_true", help="Print extra details (default: False)")
parser.add_argument("-C", "--checkpoint-path", type=str, help="The path to a checkpoint file for the model (optional)")
parser.add_argument("-t", "--tokenizer-path", type=str, help="The path to a tokenizer file for the model, in .json or .model format. If not provided, looks in the model path. (optional)")

args = parser.parse_args()

host = args.host
port = args.port
model_path = args.model_path
batch_size = args.batch_size
max_chunk_size = args.max_chunk_size
verbose = args.verbose
checkpoint_path = args.checkpoint_path
tokenizer_path = args.tokenizer_path

server = IllamaServer(
    host=host,
    port=port,
    model_path=model_path,
    batch_size=batch_size,
    max_chunk_size=max_chunk_size,
    verbose=verbose,
    checkpoint_path=checkpoint_path,
    tokenizer_path=tokenizer_path
)
server.serve()