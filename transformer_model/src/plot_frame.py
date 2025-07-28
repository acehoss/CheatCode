import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
from datetime import datetime, timedelta
from dataset import CheatCodeDataset, DataFrame
import argparse
from matplotlib.widgets import Button  # Import Button widget
import torch
import os
from cheat_code_model import CheatCodeModel  # Import the model class
import math
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.animation as animation  # Add import
# Ensure mplfinance is installed: pip install mplfinance

def load_model(model_path, device):
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # Attempt to load model_config from initial_checkpoint.pt
        initial_checkpoint_path = os.path.join(os.path.dirname(model_path), 'initial_checkpoint.pt')
        if os.path.exists(initial_checkpoint_path):
            initial_checkpoint = torch.load(initial_checkpoint_path, map_location=device)
            if 'model_config' in initial_checkpoint:
                model_config = initial_checkpoint['model_config']
            else:
                raise KeyError("model_config not found in initial_checkpoint.pt")
        else:
            raise KeyError("model_config not found in the supplied checkpoint and initial_checkpoint.pt does not exist")

    # Create the model with the loaded configuration
    model = CheatCodeModel(**model_config)

    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.float()
    model.to(device)
    model.eval()
    print("Model loaded.")
    return model

def plot_data(frame, symbol, ax1, ax2, dubba_labels_both):
    # Use 1-minute data
    data = frame.minute_data[symbol]
    time_range = 240  # Reduced time range for better performance
    data = data[-time_range:]
    date_formatter = mdates.DateFormatter('%H:%M')
    major_locator = mdates.HourLocator(interval=1)

    timestamps = mdates.date2num([frame.timestamp - timedelta(minutes=(time_range - 1 - i))
                                  for i in range(time_range)])
    ohlc = np.column_stack([timestamps, data[:, 0], data[:, 1], data[:, 2], data[:, 3]])

    # Clear previous plots
    ax1.cla()
    ax2.cla()  # Add this line to clear ax2

    # Rest of your plotting code...

    # Plot candlestick chart
    candlestick_ohlc(ax1, ohlc, width=1/ (24*60), colorup='g', colordown='r')

    # Set x-axis format
    ax1.xaxis.set_major_formatter(date_formatter)
    ax1.xaxis.set_major_locator(major_locator)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.set_title(f'{symbol} Minute Candlestick Chart')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Use provided dubba labels
    dubba_labels = {'NQ': dubba_labels_both[0], 'ES': dubba_labels_both[1]}[symbol]
    dubba_timestamps = [frame.timestamp + i * timedelta(minutes=1) for i in range(30)]

    # Initialize total signal sums
    total_long_signal = 0
    total_short_signal = 0

    for i, timestamp in enumerate(dubba_timestamps):
        for j in range(15):  # Adjusted loop for 30 price bands
            orig_long_signal_strength = dubba_labels[i, 15 + j].item()
            orig_short_signal_strength = dubba_labels[i, 14 - j].item()
            long_signal_strength = orig_long_signal_strength
            short_signal_strength = orig_short_signal_strength
            print(f"{i:02d} {j:02d} {short_signal_strength:0.2f} {long_signal_strength:0.2f}")

            if (orig_long_signal_strength * 0.85) < orig_short_signal_strength or orig_long_signal_strength < 0.2:
                long_signal_strength = 0

            if (orig_short_signal_strength * 0.85) < orig_long_signal_strength or orig_short_signal_strength < 0.2:
                short_signal_strength = 0

            total_long_signal += orig_long_signal_strength
            total_short_signal += orig_short_signal_strength

            ax2.add_patch(Rectangle(
                (timestamp, 15 + j - 14.5),  # Adjusted y-position
                timedelta(minutes=1), 1,
                fill=True,
                color='green',
                alpha=long_signal_strength
            ))
            ax2.add_patch(Rectangle(
                (timestamp, 14 - j - 14.5),  # Adjusted y-position
                timedelta(minutes=1), 1,
                fill=True,
                color='red',
                alpha=short_signal_strength
            ))

    ax2.set_ylim(-15, 15)
    ax2.set_yticks([-15, -7.5, 0, 7.5, 15])
    ax2.set_yticklabels(['Short 15', 'Short 7', 'No Trade', 'Long 7', 'Long 15'])
    ax2.set_xlabel('Time')
    ax2.set_title('Dubba Outputs')

    # Set x-axis format for dubba plot
    ax2.xaxis.set_major_formatter(date_formatter)
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
    ax2.set_xlim(frame.timestamp, frame.timestamp + 30 * timedelta(minutes=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    sig_diff = total_long_signal - total_short_signal

    # After the loops, display the sums on the plot
    ax2.text(
        0.02, 0.95, f"{'+' if sig_diff >= 0 else '-'}{sig_diff:.1f}",
        transform=ax2.transAxes, fontsize=12, verticalalignment='top', color='g' if sig_diff > 0 else 'r'
    )

def plot_frame(dataset: CheatCodeDataset, initial_frame_number: int, symbol: str, use_model_inference=False, model_path=None):
    frame_number = initial_frame_number  # Initialize frame number

    # Variable to control automatic playback
    play_direction = [0]  # Use list for mutability in nested functions
    is_updating = [False]  # Flag to indicate whether an update is in progress

    # Create a figure and axes outside the update function
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

    if use_model_inference:
        if model_path is None:
            raise ValueError("Model path must be provided when use_model_inference is True")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_path, device)
    else:
        model = None
        device = None

    # Initial plot
    frame: DataFrame = dataset.get_data_frame(frame_number)
    if use_model_inference:
        # Prepare data for inference
        minute_data_list = []
        for sym in ['NQ', 'ES']:
            minute_data = frame.minute_data[sym].unsqueeze(0)  # Add batch dimension
            minute_data_list.append(minute_data.to(device))

        # Run inference
        with torch.no_grad():
            outputs = model(minute_data_list)
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(outputs)
            # Reshape probabilities to match expected labels shape
            dubba_labels_both = probabilities.cpu().numpy().reshape(2, 30, 30)
    else:
        dubba_labels_both = frame.labels.reshape(2, 30, 30)

    plot_data(frame, symbol, ax1, ax2, dubba_labels_both)

    # Set the initial title
    fig.suptitle(f"{symbol} Minute Chart and Dubba Outputs (Frame: {frame_number})")

    # Create a timer for automatic playback
    timer = fig.canvas.new_timer(interval=2000)  # Interval will be adjusted dynamically

    def update_plot(direction):
        if is_updating[0]:
            return  # Skip if an update is already in progress
        is_updating[0] = True
        nonlocal frame_number

        frame_number += direction
        frame_number = max(0, min(frame_number, len(dataset) - 1))

        # Get the new data frame
        frame: DataFrame = dataset.get_data_frame(frame_number)

        if use_model_inference:
            # Prepare data for inference
            minute_data_list = []
            for sym in ['NQ', 'ES']:
                minute_data = frame.minute_data[sym].unsqueeze(0)  # Add batch dimension
                minute_data_list.append(minute_data.to(device))

            # Run inference
            with torch.no_grad():
                outputs = model(minute_data_list)
                probabilities = torch.sigmoid(outputs)
                dubba_labels_both = probabilities.cpu().numpy().reshape(2, 30, 30)
        else:
            dubba_labels_both = frame.labels.reshape(2, 30, 30)

        # Update the plots
        plot_data(frame, symbol, ax1, ax2, dubba_labels_both)
        fig.suptitle(f"{symbol} Minute Chart and Dubba Outputs (Frame: {frame_number})")
        fig.canvas.draw_idle()

        is_updating[0] = False  # Mark the update as completed

        # If in play mode, start the timer for the next update
        if play_direction[0] != 0:
            timer.start()

    def on_timer():
        if play_direction[0] != 0 and not is_updating[0]:
            update_plot(play_direction[0])

    # Configure the timer
    timer.add_callback(on_timer)

    def on_play_forward_clicked(event):
        if play_direction[0] == 1:
            play_direction[0] = 0  # Stop playing forward
            btn_playfwd.label.set_text('Play Forward')
        else:
            play_direction[0] = 1  # Start playing forward
            btn_playfwd.label.set_text('Stop')
            btn_playbwd.label.set_text('Play Backward')
            if not is_updating[0]:
                timer.start()

    def on_play_backward_clicked(event):
        if play_direction[0] == -1:
            play_direction[0] = 0  # Stop playing backward
            btn_playbwd.label.set_text('Play Backward')
        else:
            play_direction[0] = -1  # Start playing backward
            btn_playbwd.label.set_text('Stop')
            btn_playfwd.label.set_text('Play Forward')
            if not is_updating[0]:
                timer.start()

    # Add buttons for navigation
    axprev = plt.axes([0.7, 0.01, 0.05, 0.05])
    axnext = plt.axes([0.76, 0.01, 0.05, 0.05])
    axprev50 = plt.axes([0.82, 0.01, 0.05, 0.05])
    axnext50 = plt.axes([0.88, 0.01, 0.05, 0.05])
    axplaybwd = plt.axes([0.58, 0.01, 0.1, 0.05])
    axplayfwd = plt.axes([0.46, 0.01, 0.1, 0.05])
    btn_prev = Button(axprev, '<')
    btn_next = Button(axnext, '>')
    btn_prev50 = Button(axprev50, '<<')
    btn_next50 = Button(axnext50, '>>')
    btn_playbwd = Button(axplaybwd, 'Play Backward')
    btn_playfwd = Button(axplayfwd, 'Play Forward')

    btn_prev.on_clicked(lambda event: update_plot(-1))
    btn_next.on_clicked(lambda event: update_plot(1))
    btn_prev50.on_clicked(lambda event: update_plot(-50))
    btn_next50.on_clicked(lambda event: update_plot(50))
    btn_playbwd.on_clicked(on_play_backward_clicked)
    btn_playfwd.on_clicked(on_play_forward_clicked)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a frame from the CheatCodeDataset")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--frame_period", type=int, default=60, help="Frame period in seconds")
    parser.add_argument("--frame_number", type=int, required=True, help="Frame number to plot")
    parser.add_argument("--symbol", type=str, choices=['NQ', 'ES'], required=True, help="Symbol to plot (NQ or ES)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model checkpoint")
    parser.add_argument("--dataset_file", type=str, default=None, help="Path to the dataset file")
    
    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    if args.dataset_file is not None:
        print(f"Loading dataset from file {args.dataset_file}")
        dataset = CheatCodeDataset.from_file(args.dataset_file)
        # Adjust dates if necessary
        if dataset.start_date > start_date - timedelta(days=1):
            start_date = dataset.start_date + timedelta(days=1)
            print(f"Adjusted start date to {start_date} due to dataset")
        if dataset.end_date < end_date + timedelta(minutes=30):
            end_date = dataset.end_date - timedelta(minutes=30)
            print(f"Adjusted end date to {end_date} due to dataset")
    else:
        db_params = {
            "dbname": "trade_data",
            "user": "postgres",
            "password": "mysecretpassword",
            "host": "localhost",
            "port": "5432"
        }
        dataset = CheatCodeDataset(start_date, end_date, db_params, args.frame_period)

    use_model_inference = args.model_path is not None

    # Plot the specified frame
    plot_frame(
        dataset,
        args.frame_number,
        args.symbol,
        use_model_inference=use_model_inference,
        model_path=args.model_path
    )