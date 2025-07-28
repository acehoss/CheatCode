import os
import json
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone

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

def nanoseconds_to_timestamp(nanoseconds):
    return datetime.fromtimestamp(nanoseconds / 1e9, tz=timezone.utc)

def raw_to_decimal_price(price_raw):
    return int(price_raw) / 1e9

def process_file(file_path, conn):
    cursor = conn.cursor()
    try:
        with open(file_path, 'r') as file:
            data = []
            for line in file:
                record = json.loads(line)
                data.append((
                    nanoseconds_to_timestamp(int(record['hd']['ts_event'])),
                    60,  # Assuming 1-minute candlesticks, adjust as needed
                    record['hd']['instrument_id'],
                    raw_to_decimal_price(record['open']),
                    raw_to_decimal_price(record['high']),
                    raw_to_decimal_price(record['low']),
                    raw_to_decimal_price(record['close']),
                    record['volume']
                ))
            
            execute_values(cursor, """
                INSERT INTO candlestick_data (
                    timestamp, duration_seconds, instrument_id,
                    open_price, high_price, low_price, close_price, volume
                ) VALUES %s
            """, data)
        
        conn.commit()
        print(f"File {file_path} processed and committed successfully.")
    except Exception as e:
        conn.rollback()
        print(f"An error occurred while processing {file_path}: {e}")
    finally:
        cursor.close()

def main():
    conn = connect_to_db()

    try:
        for file_name in os.listdir('.'):
            if file_name.endswith('.jsonl'):
                print(f"Processing file: {file_name}")
                process_file(file_name, conn)
        
        print("All files processed.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()