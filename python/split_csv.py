import os
import pandas as pd
from pathlib import Path

def split_csv(input_file, chunk_size_mb=29, output_dir=None):
    """
    Split a large CSV file into smaller chunks of specified size while preserving headers.
    
    Parameters:
    input_file (str): Path to the input CSV file
    chunk_size_mb (int): Maximum size of each chunk in megabytes (default: 29MB)
    output_dir (str): Directory to save the chunks (default: same as input file)
    
    Returns:
    list: List of paths to the created chunk files
    """
    
    # Convert chunk size to bytes (with some buffer room)
    chunk_size = chunk_size_mb * 1000000  # Using 1000000 instead of 1024*1024 for safety margin
    
    # Get input file path and name
    input_path = Path(input_file)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = input_path.stem
    
    # Initialize chunk counter and output file list
    chunk_number = 1
    output_files = []
    
    try:
        # Verify file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Could not find input file: {input_file}")
            
        # Read the CSV in chunks
        for chunk in pd.read_csv(input_file, chunksize=10000):  # Adjust chunksize as needed
            current_chunk_size = 0
            current_chunk_data = []
            
            for _, row in chunk.iterrows():
                # Convert row to string and get its size
                row_size = len(row.to_string().encode('utf-8'))
                
                # If adding this row would exceed chunk size, write current chunk
                if current_chunk_size + row_size > chunk_size and current_chunk_data:
                    # Create output filename
                    output_file = output_dir / f"{base_name}_chunk_{chunk_number}.csv"
                    
                    # Convert to DataFrame and save
                    pd.DataFrame(current_chunk_data, columns=chunk.columns).to_csv(
                        output_file, index=False
                    )
                    
                    output_files.append(output_file)
                    chunk_number += 1
                    current_chunk_data = []
                    current_chunk_size = 0
                
                # Add row to current chunk
                current_chunk_data.append(row)
                current_chunk_size += row_size
            
            # Write remaining data in this iteration
            if current_chunk_data:
                output_file = output_dir / f"{base_name}_chunk_{chunk_number}.csv"
                pd.DataFrame(current_chunk_data, columns=chunk.columns).to_csv(
                    output_file, index=False
                )
                output_files.append(output_file)
                
        print(f"\nSplit complete! Created {len(output_files)} chunks:")
        for f in output_files:
            size_mb = os.path.getsize(f) / 1000000
            print(f"- {f.name}: {size_mb:.2f}MB")
        
        return output_files
    
    except Exception as e:
        print(f"\nError splitting file: {str(e)}")
        return []

if __name__ == "__main__":
    print("CSV File Splitter")
    print("-----------------")
    
    # Get input file path
    while True:
        input_csv = input("\nEnter the path to your CSV file: ").strip()
        if input_csv:
            # Remove quotes if user copied a path with quotes
            input_csv = input_csv.strip('"\'')
            if os.path.exists(input_csv):
                break
            else:
                print(f"Error: File not found at '{input_csv}'")
        else:
            print("Please enter a valid file path")
    
    # Get output directory
    while True:
        output_directory = input("\nEnter output directory path (press Enter for same as input file): ").strip()
        if not output_directory:
            output_directory = None
            break
        
        # Remove quotes if user copied a path with quotes
        output_directory = output_directory.strip('"\'')
        
        try:
            # Try to create directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)
            break
        except Exception as e:
            print(f"Error creating directory: {str(e)}")
    
    # Get chunk size (optional)
    while True:
        chunk_size_input = input("\nEnter maximum chunk size in MB (press Enter for default 29MB): ").strip()
        if not chunk_size_input:
            chunk_size_mb = 29
            break
        try:
            chunk_size_mb = float(chunk_size_input)
            if chunk_size_mb > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\nProcessing file: {input_csv}")
    print(f"Output directory: {output_directory or 'Same as input file'}")
    print(f"Chunk size: {chunk_size_mb}MB")
    print("\nSplitting file...")
    
    split_csv(input_csv, chunk_size_mb, output_directory)