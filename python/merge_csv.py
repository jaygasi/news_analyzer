import pandas as pd
import os
from datetime import datetime

# Define the constant path for the total trading data file
TOTAL_FILE_PATH = r"C:\Users\JayHy\OneDrive\OneNote\file1.csv"

def merge_trading_data(new_file_path):
    """
    Merge a new trading CSV file with the total trading data file,
    removing any duplicates and keeping the latest entries.
    
    Args:
        new_file_path (str): Path to the new CSV file to merge
    """
    try:
        # Read the new CSV file
        new_data = pd.read_csv(new_file_path)
        print(f"Successfully read new file with {len(new_data)} rows")
        
        # Check if total file exists
        if os.path.exists(TOTAL_FILE_PATH):
            total_data = pd.read_csv(TOTAL_FILE_PATH)
            print(f"Successfully read existing total file with {len(total_data)} rows")
            
            # Combine the dataframes
            merged_df = pd.concat([total_data, new_data], ignore_index=True)
            
            # Remove duplicates keeping the last occurrence (latest version)
            merged_df = merged_df.drop_duplicates(keep='last')
            
        else:
            print("No existing total file found. Creating new file.")
            merged_df = new_data
        
        # Save the merged data
        merged_df.to_csv(TOTAL_FILE_PATH, index=False)
        print(f"\nMerged data saved to: {TOTAL_FILE_PATH}")
        print(f"Total rows in merged file: {len(merged_df)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    print("Trading Data Merger")
    print("-----------------")
    
    # Get only the new file path from user and clean it
    new_file = input("Enter the path to the new CSV file to merge: ").strip().strip('"').strip("'")
    
    # Validate files
    if not os.path.exists(new_file):
        print(f"Error: New file does not exist at path: {new_file}")
        return
        
    # Create directory for total file if it doesn't exist
    total_dir = os.path.dirname(TOTAL_FILE_PATH)
    if total_dir and not os.path.exists(total_dir):
        os.makedirs(total_dir)
    
    # Perform the merge
    merge_trading_data(new_file)

if __name__ == "__main__":
    main()