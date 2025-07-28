import torch
from torch.utils.data import Dataset
import psycopg2
from psycopg2.extras import DictCursor
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from typing import List, Dict
import h5py  # Add this import for handling HDF5 files
import pickle  # For saving and loading metadata
import os  # For file path operations
from tqdm import tqdm  # Add this import at the top of the file
import concurrent.futures
import queue
import threading

logging.basicConfig(level=logging.INFO)

@dataclass
class DataFrame:
    timestamp: datetime
    minute_data: Dict[str, torch.Tensor]
    labels: torch.Tensor

class CheatCodeDataset(Dataset):
    dubba_ratio = 0.0  # Class variable to store the dubba ratio

    def __init__(self, start_date=None, end_date=None, db_params=None, frame_period=None, 
                 data_file=None, indices=None, seed=None):
        self.data_file = data_file
        self.indices = indices
        self.seed = seed
        
        if data_file is not None:
            # Open the file temporarily to read metadata
            with h5py.File(data_file, 'r') as h5f:
                self.total_frames = h5f['data'].shape[0]
                self.start_date = datetime.fromisoformat(h5f.attrs['start_date'])
                self.end_date = datetime.fromisoformat(h5f.attrs['end_date'])
            self.product_symbols = ['NQ', 'ES']
        else:
            # Initialize for database access
            self.start_date = start_date
            self.end_date = end_date
            self.db_params = db_params
            self.product_symbols = ['NQ', 'ES']
            self.frame_period = frame_period
            self.total_frames = int((end_date - start_date).total_seconds() / frame_period)
            self.indices = None  # To be set during reindexing
            self.seed = None  # To be set during shuffling

    def __len__(self):
        return self.total_frames

    def connect_to_db(self):
        return psycopg2.connect(**self.db_params)

    def fetch_most_recent_close(self, conn, product_symbol, timestamp):
        cursor = conn.cursor(cursor_factory=DictCursor)
        query = """
        SELECT close_price
        FROM candlestick_data
        WHERE product_symbol = %s AND duration_seconds = 60 AND timestamp <= %s
        ORDER BY timestamp DESC
        LIMIT 1
        """
        cursor.execute(query, (product_symbol, timestamp))
        result = cursor.fetchone()
        cursor.close()
        if result:
            return result['close_price']
        else:
            logging.warning(f"No close price found for {product_symbol} at or before {timestamp}")
            return None

    def fetch_most_recent_open(self, conn, product_symbol, timestamp):
        cursor = conn.cursor(cursor_factory=DictCursor)
        query = """
        SELECT open_price
        FROM candlestick_data
        WHERE product_symbol = %s AND duration_seconds = 60 AND timestamp >= %s
        ORDER BY timestamp ASC
        LIMIT 1
        """
        cursor.execute(query, (product_symbol, timestamp))
        result = cursor.fetchone()
        cursor.close()
        if result:
            return result['open_price']
        else:
            logging.warning(f"No open price found for {product_symbol} at or after {timestamp}")
            return None

    def __getitem__(self, idx):
        if self.data_file is not None:
            # Open the file for each __getitem__ call
            with h5py.File(self.data_file, 'r') as h5f:
                data_ds = h5f['data']
                labels_ds = h5f['labels']
                
                idx = self.indices[idx] if self.indices is not None else idx
                data_array = data_ds[idx]
                labels_array = labels_ds[idx]
                
                minute_data = [torch.tensor(data_array[i], dtype=torch.float32) for i in range(len(data_array))]
                labels = torch.tensor(labels_array, dtype=torch.float32)
                
                # Check for NaNs/Infs in minute_data
                for data in minute_data:
                    if not torch.isfinite(data).all():
                        raise ValueError(f"Infinite or NaN values found in minute_data at index {idx}")

                # Check for NaNs/Infs in labels
                if not torch.isfinite(labels).all():
                    raise ValueError(f"Infinite or NaN values found in labels at index {idx}")

                return {
                    'minute_data': minute_data,
                    'labels': labels
                }
        else:
            # Existing code to fetch data from the database
            idx = int(idx)
            current_date = self.start_date + timedelta(seconds=idx * self.frame_period)
            all_data = []
            labels = []

            with self.connect_to_db() as conn:
                for product_symbol in self.product_symbols:
                    # Only fetch 1-minute candles
                    start_time = current_date - timedelta(days=1)
                    candles = self.fetch_candlestick_data(conn, product_symbol, start_time, current_date, 60)
                    candle_data = self.process_candles(candles, 60)
                    all_data.append(candle_data)
                    labels.extend(self.generate_dubba_labels(conn, product_symbol, current_date, candles[-1]['absolute_close']))

            # After generating all labels, compute the dubba ratio
            all_labels = torch.tensor(labels, dtype=torch.float32)
            dubba_count = torch.sum(all_labels)
            total_count = all_labels.numel()
            
            # Update the class variable with the new ratio
            CheatCodeDataset.dubba_ratio = dubba_count.item() / total_count * 0.1 + CheatCodeDataset.dubba_ratio * 0.9

            # Restructure the data for return
            return {
                'minute_data': [torch.tensor(data, dtype=torch.float32) for data in all_data],
                'labels': all_labels
            }

    def fetch_candlestick_data(self, conn, product_symbol, start_time, end_time, duration_seconds, close_date=None, compare_to_price=None):
        cursor = conn.cursor(cursor_factory=DictCursor)
        query = """
        SELECT timestamp, open_price, high_price, low_price, close_price, volume
        FROM candlestick_data
        WHERE product_symbol = %s AND duration_seconds = %s AND timestamp >= %s AND timestamp < %s
        ORDER BY timestamp
        """
        cursor.execute(query, (product_symbol, duration_seconds, start_time, end_time))
        fetched_candles = cursor.fetchall()
        cursor.close()
        
        # Calculate the expected number of candles
        expected_candles = int((end_time - start_time).total_seconds() / duration_seconds)
        
        filled_candles = []
        current_time = start_time
        fetched_index = 0

        # Fetch the close price of the whole period
        period_close_price = compare_to_price or float(self.fetch_most_recent_close(conn, product_symbol, close_date or end_time))
        if period_close_price is None:
            period_close_price = float(self.fetch_most_recent_open(conn, product_symbol, close_date or end_time))
            if period_close_price is None:
                period_close_price = 1.0  # Default to 1.0 if no close price found

        while current_time < end_time:
            if fetched_index < len(fetched_candles) and fetched_candles[fetched_index]['timestamp'] == current_time:
                candle = fetched_candles[fetched_index]
                absolute_open = float(candle['open_price'])
                absolute_high = float(candle['high_price'])
                absolute_low = float(candle['low_price'])
                absolute_close = float(candle['close_price'])
                
                filled_candles.append({
                    'timestamp': candle['timestamp'],
                    'open_price': absolute_open / period_close_price,
                    'high_price': absolute_high / period_close_price,
                    'low_price': absolute_low / period_close_price,
                    'close_price': absolute_close / period_close_price,
                    'volume': float(candle['volume']),
                    'absolute_open': absolute_open,
                    'absolute_close': absolute_close
                })
                
                fetched_index += 1
            else:
                # Fill in with a zero volume point price candlestick
                last_price = self.fetch_most_recent_close(conn, product_symbol, current_time)
                if last_price is None:
                    # If no past close price, try to get the nearest future open price
                    last_price = self.fetch_most_recent_open(conn, product_symbol, current_time)
                    if last_price is None:
                        # If still no price, use the period close price
                        last_price = period_close_price
                
                filled_candles.append({
                    'timestamp': current_time,
                    'open_price': float(last_price) / period_close_price,
                    'high_price': float(last_price) / period_close_price,
                    'low_price': float(last_price) / period_close_price,
                    'close_price': float(last_price) / period_close_price,
                    'volume': 0,
                    'absolute_open': float(last_price),
                    'absolute_close': float(last_price)
                })
            
            current_time += timedelta(seconds=duration_seconds)
        
        return filled_candles

    def process_candles(self, candles, duration):
        processed_data = []
        for candle in candles:
            timestamp = candle['timestamp']
            open_price = float(candle['open_price'])
            high_price = float(candle['high_price'])
            low_price = float(candle['low_price'])
            close_price = float(candle['close_price'])
            volume = float(candle['volume'])
            sinusoidal_embeddings = self.calculate_sinusoidal_embeddings(timestamp, duration)
            processed_data.append([open_price, high_price, low_price, 
                                   close_price, volume] + sinusoidal_embeddings)
        return np.array(processed_data, dtype=float)

    def calculate_sinusoidal_embeddings(self, timestamp, duration):
        time_of_day = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        day_of_week = timestamp.weekday()
        day_of_month = timestamp.day - 1
        day_of_year = timestamp.timetuple().tm_yday - 1

        def get_sinusoidal_embedding(value, max_value):
            return [np.sin(2 * np.pi * value / max_value), np.cos(2 * np.pi * value / max_value)]

        embeddings = (
            get_sinusoidal_embedding(time_of_day, 86400) +
            get_sinusoidal_embedding(day_of_week, 7) +
            get_sinusoidal_embedding(day_of_month, 31) +
            get_sinusoidal_embedding(day_of_year, 366)
        )

        # Add minute of the hour embedding for 1-minute and 1-second candles
        if duration in [1, 60]:
            minute_of_hour = timestamp.minute
            embeddings.extend(get_sinusoidal_embedding(minute_of_hour, 60))

        return embeddings

    def generate_dubba_labels(self, conn, product_symbol, current_date, current_price):
        labels = []
        window_duration = 1800  # Next 30 minutes
        window_slices = 30  # Every 1 minute
        labels.extend(self.generate_dubba_labels_for_window(
            conn, product_symbol, current_date, current_price, window_duration, window_slices
        ).flatten())
        return labels

    def generate_dubba_labels_for_window(self, conn, product_symbol, current_date, current_price, window_duration, window_slices):
        num_price_bands = 30
        labels = np.zeros((window_slices, 30))
        
        # Define tick size for each product
        tick_sizes = {'NQ': 2.0, 'ES': 2.0}  # You may need to adjust these values
        tick_size = tick_sizes[product_symbol]
        
        # Fetch future candles for the entire window
        end_time = current_date + timedelta(seconds=window_duration)
        future_candles = self.fetch_candlestick_data(conn, product_symbol, current_date, end_time, 60, current_date, current_price)
        
        # Calculate relative tick size
        relative_tick_size = tick_size / current_price
        
        # The prices are already relative, so we don't need to divide by current_price here
        future_prices_max = np.array([float(candle['high_price']) for candle in future_candles])
        future_prices_min = np.array([float(candle['low_price']) for candle in future_candles])
        
        slice_duration = window_duration // window_slices
        candles_per_slice = 60 // slice_duration # since we're using 1min candles
        
        for slice_idx in range(window_slices):
            start_idx = 0 # must be no movement against us between now and dubba
            end_idx = int((slice_idx + 1) * candles_per_slice)
            slice_prices_max = future_prices_max[start_idx:end_idx]
            slice_prices_min = future_prices_min[start_idx:end_idx]
            
            max_price = np.max(slice_prices_max)
            min_price = np.min(slice_prices_min)
            for j in range(num_price_bands//2):
                price_movement = (j + 1) * relative_tick_size
                drawdown_limit = price_movement / 2
                
                long_target_price = 1 + price_movement
                long_drawdown_limit_price = 1 - drawdown_limit
                short_target_price = 1 - price_movement
                short_drawdown_limit_price = 1 + drawdown_limit
                
                # Long dubba
                if max_price >= long_target_price and min_price >= long_drawdown_limit_price:
                    labels[slice_idx, num_price_bands//2 + j] = 1
                
                # Short dubba
                if min_price <= short_target_price and max_price <= short_drawdown_limit_price:
                    labels[slice_idx, num_price_bands//2 - 1 - j] = 1
        
        return labels

    def get_data_frame(self, idx: int) -> DataFrame:
        """
        Retrieves a DataFrame object for the given index, representing a specific point in time.

        Parameters:
        - idx (int): The index of the desired data frame.

        Returns:
        - DataFrame: An object containing the following attributes:

          1. timestamp (datetime): The exact date and time this frame represents.

          2. minute_data (Dict[str, torch.Tensor]): 
             Contains 1-minute candlestick data for the past day.
             Keys: 'NQ', 'ES' (product symbols)
             Values: torch.Tensor of shape (1440, 15), where:
               - 1440 represents the number of minutes in a day
               - 15 is the number of features per candle:
                 [0] open_price (relative to period close)
                 [1] high_price (relative to period close)
                 [2] low_price (relative to period close)
                 [3] close_price (relative to period close)
                 [4] volume
                 [5-14] sinusoidal time embeddings (10 values, see note on embeddings)

          3. labels (torch.Tensor): 
             A tensor of shape (15000,) representing the dubba labels for both tickers.
             Each ticker symbol has a set of 30 time slices, and within each slice, each time slice has 250 1-tick price bins (125 long and 125 short).
             The time slices are organized as follows:
             - 7500 values: 30 slices of 1-minute windows (30 * 250 = 7500)
             Each slice contains 125 short labels followed by 125 long labels. Lower indexes mean lower price.
             Label 1 indicates a profitable trade, 0 indicates no trade or unprofitable trade.

        Note on sinusoidal time embeddings:
        The embeddings encode temporal information using sine and cosine functions:
        - For minute_data (10 values):
          [0-1]: Time of day (sin, cos) - Period: 24 hours
          [2-3]: Day of week (sin, cos) - Period: 7 days
          [4-5]: Day of month (sin, cos) - Period: 31 days
          [6-7]: Day of year (sin, cos) - Period: 366 days
          [8-9]: Minute of hour (sin, cos) - Period: 60 minutes

        Each pair of values (sin, cos) represents a point on a unit circle:
        - The angle of this point corresponds to the position within the period
        - sin(0) = 0, cos(0) = 1 represents the start of the period
        - sin(π/2) = 1, cos(π/2) = 0 represents 1/4 through the period
        - sin(π) = 0, cos(π) = -1 represents halfway through the period
        - sin(3π/2) = -1, cos(3π/2) = 0 represents 3/4 through the period

        To interpret:
        - atan2(cos, sin) gives the angle in radians
        - (atan2(cos, sin) + π) / (2π) gives the fraction of the period elapsed

        These embeddings allow the model to understand cyclical patterns in time
        and can be used by the plotter to accurately position data points within
        their respective time cycles.

        Note:
        - All price data (open, high, low, close) is relative to the period close price.
        - The dubba labels represent potential profitable trades at various time scales and price movements.
        """
        idx = int(idx)
        current_date = self.start_date + timedelta(seconds=idx * self.frame_period)
        data = self.__getitem__(idx)
        
        return DataFrame(
            timestamp=current_date,
            minute_data={
                'NQ': data['minute_data'][0],
                'ES': data['minute_data'][1]
            },
            labels=data['labels']
        )

    def save_to_file(self, file_path, indices=None, num_workers=64, batch_size=256):
        """
        Saves the dataset to an HDF5 file specified by file_path.
        If indices are provided, data is reordered accordingly.
        Supports resuming from where it left off if the process was interrupted.
        """
        print(f"Saving dataset to {file_path}")

        # Check if the file exists and resume if possible
        if os.path.exists(file_path):
            print(f"Resuming from existing file {file_path}")
            h5f = h5py.File(file_path, 'a')
            data_ds = h5f['data']
            labels_ds = h5f['labels']
            num_samples = data_ds.shape[0]
            # Get last saved index from file attribute; default to 0 if not set
            last_saved_idx = h5f.attrs.get('last_saved_idx', 0)
        else:
            print(f"Creating new file {file_path}")
            # Open the HDF5 file
            h5f = h5py.File(file_path, 'w')
            # Create datasets for data and labels with appropriate shapes
            num_samples = len(self)
            data_shape = (num_samples, len(self.product_symbols), 1440, 15)  # Adjust dimensions as needed
            label_shape = (num_samples,) + self.__getitem__(0)['labels'].shape

            # Calculate optimal chunk size (adjust as needed)
            chunk_size = (1, len(self.product_symbols), 1440, 15)
            label_chunk_size = (1,) + label_shape[1:]

            data_ds = h5f.create_dataset('data', shape=data_shape, dtype='float32', 
                                         chunks=chunk_size, compression='lzf')
            labels_ds = h5f.create_dataset('labels', shape=label_shape, dtype='float32', 
                                           chunks=label_chunk_size, compression='lzf')

            # Save metadata as attributes
            h5f.attrs['start_date'] = self.start_date.isoformat()
            h5f.attrs['end_date'] = self.end_date.isoformat()
            h5f.attrs['seed'] = self.seed if self.seed is not None else -1

            # Initialize last_saved_idx to 0
            last_saved_idx = 0

        # Calculate number of batches starting from last_saved_idx
        num_samples_remaining = num_samples - last_saved_idx
        num_batches = (num_samples_remaining + batch_size - 1) // batch_size

        # Create a progress bar, starting from last_saved_idx
        pbar = tqdm(total=num_samples, initial=last_saved_idx, desc="Fetching samples", unit="sample")

        # **Add this line to initialize batch_queue**
        batch_queue = queue.Queue(maxsize=2)  # Initialize the batch_queue

        # Function to fetch a single item
        def fetch_item(idx):
            actual_idx = indices[idx] if indices is not None else idx
            item = self.__getitem__(actual_idx)
            pbar.update(1)
            return item

        # Producer function to fetch batches starting from last_saved_idx
        def producer():
            for batch_start_idx in range(last_saved_idx, num_samples, batch_size):
                end_idx = min(batch_start_idx + batch_size, num_samples)
                batch_indices = range(batch_start_idx, end_idx)

                # Fetch items in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    batch_data = list(executor.map(fetch_item, batch_indices))

                # Prepare batch data and labels
                data_array = np.array([
                    [d.numpy() for d in sample['minute_data']]
                    for sample in batch_data
                ])
                labels_array = np.array([sample['labels'].numpy() for sample in batch_data])

                batch_queue.put((batch_start_idx, data_array, labels_array))

            # Signal that all batches have been produced
            batch_queue.put(None)
            pbar.close()

        # Consumer function to write batches to file
        def consumer():
            while True:
                batch = batch_queue.get()
                if batch is None:  # End signal
                    break
                start_idx, data_array, labels_array = batch
                print(f"Writing batch, start_idx: {start_idx}")
                # Write batch data and labels
                end_idx = min(start_idx + batch_size, num_samples)
                data_ds[start_idx:end_idx] = data_array
                labels_ds[start_idx:end_idx] = labels_array

                # After writing batch, update last_saved_idx and flush
                h5f.attrs['last_saved_idx'] = end_idx
                h5f.flush()

                batch_queue.task_done()

        # Start producer and consumer threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        consumer_thread.start()

        # Wait for both threads to finish
        producer_thread.join()
        consumer_thread.join()

        print("All samples saved.")

        # Save indices if available
        if indices is not None:
            if 'indices' in h5f:
                del h5f['indices']  # Delete existing indices dataset
            h5f.create_dataset('indices', data=indices, compression='lzf')
            print("Indices saved.")

        # Close the HDF5 file
        h5f.close()

    def reindex(self, indices):
        # Simply store the indices
        self.indices = indices
        return self

    @classmethod
    def from_file(cls, file_path):
        return cls(data_file=file_path)