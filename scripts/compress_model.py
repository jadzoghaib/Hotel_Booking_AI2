import joblib
import os
import gzip

try:
    print("Loading model...")
    model = joblib.load('best_model.pkl')
    
    print("Compressing model (level 3)...")
    joblib.dump(model, 'best_model_compressed.pkl', compress=3)
    
    original_size = os.path.getsize('best_model.pkl') / (1024 * 1024)
    compressed_size = os.path.getsize('best_model_compressed.pkl') / (1024 * 1024)
    
    print(f"Original size: {original_size:.2f} MB")
    print(f"Compressed size: {compressed_size:.2f} MB")
    
except Exception as e:
    print(f"Error: {e}")
