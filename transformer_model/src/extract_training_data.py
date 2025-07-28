import psycopg2
from psycopg2.extras import DictCursor
from decimal import Decimal
import numpy as np
from datetime import datetime, timedelta
import random
import argparse

# Database connection parameters
DB_NAME = "trade_data"
DB_USER = "postgres"
DB_PASSWORD = "mysecretpassword"
DB_HOST = "localhost"
DB_PORT = "5432"

def connect_to_db():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def fetch_candlestick_data(conn, product_symbol, start_time, end_time, duration_seconds):
    cursor = conn.cursor(cursor_factory=DictCursor)
    
    # First, fetch the most recent candle before the start time
    recent_candle_query = """
    SELECT close_price
    FROM candlestick_data
    WHERE product_symbol = %s AND timestamp < %s AND duration_seconds = %s
    ORDER BY timestamp DESC
    LIMIT 1
    """
    cursor.execute(recent_candle_query, (product_symbol, start_time, duration_seconds))
    recent_candle = cursor.fetchone()
    
    initial_close = float(recent_candle['close_price']) if recent_candle else None

    # Fetch the candles for the specified time range
    query = """
    SELECT timestamp, open_price, high_price, low_price, close_price, volume, vwap
    FROM candlestick_data
    WHERE product_symbol = %s AND timestamp >= %s AND timestamp < %s AND duration_seconds = %s
    ORDER BY timestamp
    """
    cursor.execute(query, (product_symbol, start_time, end_time, duration_seconds))
    raw_data = cursor.fetchall()

    filled_data = []
    expected_timestamp = start_time
    prev_close = initial_close
    zero_volume_candles_added = 0

    for candle in raw_data:
        timestamp = candle['timestamp']
        
        while expected_timestamp < timestamp:
            if prev_close is not None:
                filled_data.append((expected_timestamp, float(prev_close), float(prev_close), float(prev_close), float(prev_close), 0, float(prev_close)))
                zero_volume_candles_added += 1
            expected_timestamp += timedelta(seconds=duration_seconds)

        filled_data.append((
            timestamp,
            float(candle['open_price']),
            float(candle['high_price']),
            float(candle['low_price']),
            float(candle['close_price']),
            float(candle['volume']),
            float(candle['vwap'])
        ))
        expected_timestamp = timestamp + timedelta(seconds=duration_seconds)
        prev_close = candle['close_price']

    while expected_timestamp < end_time:
        if prev_close is not None:
            filled_data.append((expected_timestamp, float(prev_close), float(prev_close), float(prev_close), float(prev_close), 0, float(prev_close)))
            # zero_volume_candles_added += 1
        expected_timestamp += timedelta(seconds=duration_seconds)

    # if zero_volume_candles_added > 0:
        # print(f"  Added {zero_volume_candles_added} zero volume candles for {product_symbol} @ {duration_seconds}s")

    return filled_data

def calculate_sinusoidal_embeddings(timestamp):
    # Calculate sinusoidal embeddings for time features
    time_of_day = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    day_of_week = timestamp.weekday()
    day_of_month = timestamp.day - 1
    day_of_year = timestamp.timetuple().tm_yday - 1

    def get_sinusoidal_embedding(value, max_value):
        return [np.sin(2 * np.pi * value / max_value), np.cos(2 * np.pi * value / max_value)]

    return (
        get_sinusoidal_embedding(time_of_day, 86400) +
        get_sinusoidal_embedding(day_of_week, 7) +
        get_sinusoidal_embedding(day_of_month, 31) +
        get_sinusoidal_embedding(day_of_year, 366)
    )

def generate_dubba_labels(price_data, window_size):
    labels = np.zeros((len(price_data), 250))
    tick_size = 0.25
    for i in range(len(price_data) - window_size):
        current_price = price_data[i]
        future_prices = price_data[i+1:i+window_size+1]
        max_price = np.max(future_prices)
        min_price = np.min(future_prices)
        
        for j in range(125):
            ticks_away = j + 1
            price_movement = ticks_away * tick_size
            drawdown_limit = price_movement / 2

            # Long dubba
            long_target_price = current_price + price_movement
            long_drawdown_limit_price = current_price - drawdown_limit
            if max_price >= long_target_price and min_price >= long_drawdown_limit_price:
                labels[i, 125 + j] = 1
            
            # Short dubba
            short_target_price = current_price - price_movement
            short_drawdown_limit_price = current_price + drawdown_limit
            if min_price <= short_target_price and max_price <= short_drawdown_limit_price:
                labels[i, 124 - j] = 1
    
    return labels

def extract_training_data(start_date, end_date, output_file):
    conn = connect_to_db()
    
    product_symbols = ['NQ', 'ES']
    durations = [1, 60, 3600]
    
    all_data = []
    
    current_date = start_date
    while current_date < end_date:
        print(f"Processing date: {current_date}")
        
        for product_symbol in product_symbols:
            product_data = []
            
            for duration in durations:
                if duration == 1:
                    start_time = current_date - timedelta(minutes=30)
                elif duration == 60:
                    start_time = current_date - timedelta(days=1)
                else:  # 3600
                    start_time = current_date - timedelta(weeks=4)
                
                candles = fetch_candlestick_data(conn, product_symbol, start_time, current_date, duration)
                # print(f"  {product_symbol}: {len(candles)} candles @ {duration}s")

                if len(candles) == 0:
                    continue
                
                candle_data = []
                for candle in candles:
                    timestamp, open_price, high_price, low_price, close_price, volume, vwap = candle
                    sinusoidal_embeddings = calculate_sinusoidal_embeddings(timestamp)
                    candle_data.append([open_price, high_price, low_price, close_price, volume, vwap] + sinusoidal_embeddings)
                
                candle_data = np.array(candle_data, dtype=float)  # Ensure all data is float
                product_data.append(candle_data)
            
            if len(product_data) == 3:
                all_data.append(product_data)
        
        current_date += timedelta(minutes=random.randint(1, 5))
    
    conn.close()
    
    # Generate labels
    labels = []
    for product_data in all_data:
        second_data = product_data[0]
        minute_data = product_data[1]
        hour_data = product_data[2]
        
        second_labels = generate_dubba_labels(second_data[:, 3], 60)  # 1 minute (60 seconds)
        minute_labels = generate_dubba_labels(minute_data[:, 3], 10)  # 10 minutes
        hour_labels = generate_dubba_labels(hour_data[:, 3], 12)  # 1 hour (12 5-minute periods)
        
        combined_labels = np.concatenate([second_labels[-1], minute_labels[-1], hour_labels[-1]])
        labels.append(combined_labels)
    
    # Save data to file
    np.savez_compressed(
        output_file,
        data=np.array(all_data, dtype=object),
        labels=np.array(labels)
    )
    
    print(f"Training data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract training data from candlestick data")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    extract_training_data(start_date, end_date, args.output_file)