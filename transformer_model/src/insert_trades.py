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
                hd = record['hd']
                data.append((
                    record['ts_recv'],
                    hd['ts_event'],
                    nanoseconds_to_timestamp(int(hd['ts_event'])),
                    hd['rtype'],
                    hd['publisher_id'],
                    hd['instrument_id'],
                    record['action'],
                    record['side'],
                    record['depth'],
                    int(record['price']),
                    raw_to_decimal_price(record['price']),
                    record['size'],
                    record['flags'],
                    record['ts_in_delta'],
                    record['sequence']
                ))
            
            execute_values(cursor, """
                INSERT INTO trade_records (
                    ts_recv, ts_event, ts_event_datetime, rtype, publisher_id, instrument_id,
                    action, side, depth, price_raw, price, size, flags, ts_in_delta, sequence
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