import json
import sys
import psycopg2
from psycopg2 import sql
from datetime import datetime

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

def insert_instrument(conn, instrument_id, symbol, start_date, end_date):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO instruments (instrument_id, symbol, start_date, end_date)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (instrument_id) DO NOTHING
        """, (instrument_id, symbol, start_date, end_date))
        conn.commit()
        print(f"Inserted or skipped instrument: {symbol} (ID: {instrument_id})")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting instrument {symbol} (ID: {instrument_id}): {e}")
    finally:
        cursor.close()

def process_symbology_file(file_path):
    conn = connect_to_db()
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
        for symbol, instruments in data['result'].items():
            for instrument in instruments:
                instrument_id = int(instrument['s'])
                start_date = datetime.strptime(instrument['d0'], '%Y-%m-%d').date()
                end_date = datetime.strptime(instrument['d1'], '%Y-%m-%d').date()
                insert_instrument(conn, instrument_id, symbol, start_date, end_date)
        
        print(f"File {file_path} processed successfully.")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python import_symbols.py <path_to_symbology_file>")
        sys.exit(1)
    
    symbology_file_path = sys.argv[1]
    process_symbology_file(symbology_file_path)