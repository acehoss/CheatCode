import argparse
import numpy as np
import json
import sys

def load_dataset(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data['data'], data['labels']

def format_input(frame_data):
    return [
        {
            "second_data": frame_data[0][0].tolist(),
            "minute_data": frame_data[0][1].tolist(),
            "hour_data": frame_data[0][2].tolist()
        },
        {
            "second_data": frame_data[1][0].tolist(),
            "minute_data": frame_data[1][1].tolist(),
            "hour_data": frame_data[1][2].tolist()
        }
    ]

def format_output(frame_labels):
    return [
        {
            "10s": frame_labels[0:250].tolist(),
            "1m": frame_labels[250:500].tolist(),
            "5m": frame_labels[500:750].tolist()
        },
        {
            "10s": frame_labels[750:1000].tolist(),
            "1m": frame_labels[1000:1250].tolist(),
            "5m": frame_labels[1250:1500].tolist()
        }
    ]

def dump_frame(data, labels, frame_number):
    frame_data = data[frame_number]
    frame_labels = labels[frame_number]

    frame_dict = {
        "input": format_input(frame_data),
        "output": format_output(frame_labels)
    }

    with open("frame.json", "w") as f:
        json.dump(frame_dict, f, indent=2)

    print(f"Frame {frame_number} dumped to frame.json")
    

def main():
    parser = argparse.ArgumentParser(description="Dump a specific frame from a training data .npz file")
    parser.add_argument("file_path", help="Path to the training data .npz file")
    args = parser.parse_args()

    data, labels = load_dataset(args.file_path)
    num_frames = len(data)

    print(f"Number of frames in the dataset: {num_frames}")

    while True:
        try:
            frame_number = input("Enter a frame number to dump (or 'q' to quit): ")
            if frame_number.lower() == 'q':
                break

            frame_number = int(frame_number)
            if 0 <= frame_number < num_frames:
                dump_frame(data, labels, frame_number)
            else:
                print(f"Frame number must be between 0 and {num_frames - 1}")
        except ValueError:
            print("Invalid input. Please enter a valid frame number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

if __name__ == "__main__":
    main()

# Sample output of dump_frame.py:
# {
#   "input": [
#     {
#       "second_data": [1.0, 1.1, 0.9, ..., 1.2],  // 1800 values
#       "minute_data": [1.0, 1.2, 0.8, ..., 1.3],  // 1440 values
#       "hour_data": [1.0, 1.3, 0.7, ..., 1.4]     // 480 values
#     },
#     {
#       "second_data": [1.1, 1.2, 1.0, ..., 1.3],  // 1800 values
#       "minute_data": [1.1, 1.3, 0.9, ..., 1.4],  // 1440 values
#       "hour_data": [1.1, 1.4, 0.8, ..., 1.5]     // 480 values
#     }
#   ],
#   "output": [
#     {
#       "10s": [0.1, 0.2, 0.3, ..., 0.2],  // 250 values
#       "1m": [0.4, 0.5, 0.6, ..., 0.5],   // 250 values
#       "5m": [0.7, 0.8, 0.9, ..., 0.8]    // 250 values
#     },
#     {
#       "10s": [0.2, 0.3, 0.4, ..., 0.3],  // 250 values
#       "1m": [0.5, 0.6, 0.7, ..., 0.6],   // 250 values
#       "5m": [0.8, 0.9, 1.0, ..., 0.9]    // 250 values
#     }
#   ]
# }
#
# Note: The actual output will contain full-length arrays as specified above.