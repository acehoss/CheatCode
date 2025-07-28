import torch
import json
import argparse
import sys
from cheat_code_model import CheatCodeModel

def load_model(model_path, device):
    # Load the model configuration (you might need to save this during training)
    model_config = torch.load(model_path, map_location=device)['model_config']
    
    # Create the model with the saved configuration
    model = CheatCodeModel(**model_config)
    
    # Load the state dict
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def load_input_data(input_source):
    if input_source == '-':
        # Read from standard input
        return json.load(sys.stdin)
    else:
        # Read from file
        with open(input_source, 'r') as f:
            return json.load(f)

def preprocess_input(input_data):
    # Convert input data to the format expected by the model
    minute_data = [torch.tensor(ticker['minute_data'], dtype=torch.float32) for ticker in input_data]
    return minute_data

def inference(model, input_data, device):
    minute_data = preprocess_input(input_data)
    
    # Move data to the appropriate device
    minute_data = [d.to(device) for d in minute_data]
    
    with torch.no_grad():
        outputs = model(minute_data)
    
    # Convert outputs to probabilities using sigmoid
    probabilities = torch.sigmoid(outputs)
    
    return probabilities.cpu().numpy()

def format_output(probabilities):
    num_tickers = 2
    bands_per_ticker = 7500
    
    output_data = []
    
    for ticker_idx in range(num_tickers):
        ticker_data = probabilities[:, ticker_idx * bands_per_ticker : (ticker_idx + 1) * bands_per_ticker].tolist()
        output_data.append(ticker_data)
    
    return output_data

def save_output(output_data, output_destination):
    if output_destination == '-':
        # Write to standard output
        json.dump(output_data, sys.stdout, indent=2)
    else:
        # Write to file
        with open(output_destination, 'w') as f:
            json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with the CheatCode Model",
        epilog="""
Example input format:
[
  {
    "minute_data": [[...], [...], ...],  // 1440 entries, each with 15 features
  },
  {
    "minute_data": [[...], [...], ...],
  }
]

Example output format:
[
  [...],  // 7500 probabilities for ticker 1
  [...]   // 7500 probabilities for ticker 2
]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--input", type=str, default="-", help="Path to input JSON file or '-' for stdin")
    parser.add_argument("--output", type=str, default="-", help="Path to output JSON file or '-' for stdout")
    args = parser.parse_args()

    # Check for GPU availability (Nvidia CUDA or Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    # Load the model
    model = load_model(args.model, device)
    
    # Load input data
    input_data = load_input_data(args.input)
    
    # Run inference
    output_probabilities = inference(model, input_data, device)
    
    # Format output data
    output_data = format_output(output_probabilities)
    
    # Save or print output
    save_output(output_data, args.output)