import gzip
import shutil
import os

input_file = 'best_model.pkl'
output_file = 'best_model.pkl.gz'

try:
    print(f"Compressing {input_file}...")
    with open(input_file, 'rb') as f_in:
        with gzip.open(output_file, 'wb', compresslevel=9) as f_out: # Max compression
            shutil.copyfileobj(f_in, f_out)
    
    original_size = os.path.getsize(input_file) / (1024 * 1024)
    compressed_size = os.path.getsize(output_file) / (1024 * 1024)
    
    print(f"Original size: {original_size:.2f} MB")
    print(f"Compressed size: {compressed_size:.2f} MB")
    
except Exception as e:
    print(f"Error: {e}")
