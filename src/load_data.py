#!/usr/bin/env python3
"""
Telco Customer Churn Data Loader
Loads CSV data into PostgreSQL database
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from datetime import datetime

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'database': 'telco_churn',
    'user': 'anthonymerlin',
    'password': '',  # Empty password
    'port': '5432'
}

def clean_data(df):
    """Clean and prepare the data for database insertion"""
    print("Cleaning data...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Clean column names (remove spaces, convert to lowercase)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Handle 'TotalCharges' - convert to numeric (some values might be strings)
    if 'totalcharges' in df.columns:
        df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
        # Fill NaN values with 0 or monthly_charges (for new customers)
        df['totalcharges'] = df['totalcharges'].fillna(df['monthlycharges'])
    
    # Ensure senior_citizen is 0 or 1
    if 'seniorcitizen' in df.columns:
        df['seniorcitizen'] = df['seniorcitizen'].astype(int)
    
    # Clean text fields - remove extra spaces
    text_columns = ['gender', 'partner', 'dependents', 'phoneservice', 'multiplelines',
                   'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection',
                   'techsupport', 'streamingtv', 'streamingmovies', 'contract',
                   'paperlessbilling', 'paymentmethod', 'churn']
    
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Map column names to match database schema
    column_mapping = {
        'customerid': 'customer_id',
        'seniorcitizen': 'senior_citizen',
        'phoneservice': 'phone_service',
        'multiplelines': 'multiple_lines',
        'internetservice': 'internet_service',
        'onlinesecurity': 'online_security',
        'onlinebackup': 'online_backup',
        'deviceprotection': 'device_protection',
        'techsupport': 'tech_support',
        'streamingtv': 'streaming_tv',
        'streamingmovies': 'streaming_movies',
        'paperlessbilling': 'paperless_billing',
        'paymentmethod': 'payment_method',
        'monthlycharges': 'monthly_charges',
        'totalcharges': 'total_charges'
    }
    
    df = df.rename(columns=column_mapping)
    
    print(f"Data cleaned. Shape: {df.shape}")
    return df

def load_data_to_db(csv_file_path):
    """Load CSV data into PostgreSQL database"""
    
    try:
        # Read CSV file
        print(f"Reading CSV file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        print(f"CSV loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Clean the data
        df = clean_data(df)
        
        # Connect to database
        print("Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Clear existing data (optional - remove if you want to keep existing data)
        print("Clearing existing data...")
        cursor.execute("DELETE FROM customers;")
        
        # Prepare data for insertion
        columns = ['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
                  'tenure', 'phone_service', 'multiple_lines', 'internet_service',
                  'online_security', 'online_backup', 'device_protection', 'tech_support',
                  'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing',
                  'payment_method', 'monthly_charges', 'total_charges', 'churn']
        
        # Create INSERT query
        insert_query = f"""
        INSERT INTO customers ({', '.join(columns)})
        VALUES %s
        """
        
        # Convert DataFrame to list of tuples
        data_tuples = []
        for _, row in df.iterrows():
            tuple_data = tuple(row[col] if col in row and pd.notna(row[col]) else None for col in columns)
            data_tuples.append(tuple_data)
        
        # Insert data in batches
        print(f"Inserting {len(data_tuples)} records...")
        execute_values(cursor, insert_query, data_tuples, page_size=1000)
        
        # Commit the transaction
        conn.commit()
        
        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM customers;")
        count = cursor.fetchone()[0]
        print(f"✅ Successfully loaded {count} records into the database!")
        
        # Show sample data
        cursor.execute("SELECT customer_id, gender, churn, monthly_charges FROM customers LIMIT 5;")
        sample_data = cursor.fetchall()
        print("\nSample data:")
        for row in sample_data:
            print(f"  {row}")
        
        # Close connections
        cursor.close()
        conn.close()
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Error: CSV file '{csv_file_path}' not found!")
        print("Please make sure the file is in the correct location.")
        return False
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function"""
    print("🔄 Telco Customer Churn Data Loader")
    print("=" * 40)
    
    # CSV file path - adjust this to match your file location
    csv_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # Check if file exists
    import os
    if not os.path.exists(csv_file):
        print(f"❌ CSV file '{csv_file}' not found in current directory!")
        print("Please download the file from Kaggle and place it in this folder.")
        print("Current directory contents:")
        for file in os.listdir():
            print(f"  - {file}")
        return
    
    # Load the data
    success = load_data_to_db(csv_file)
    
    if success:
        print("\n🎉 Data loading completed successfully!")
        print("You can now run SQL queries on your customers table.")
    else:
        print("\n❌ Data loading failed. Please check the error messages above.")

if __name__ == "__main__":
    main()