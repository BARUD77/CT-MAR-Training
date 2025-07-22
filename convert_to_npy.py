import os
import numpy as np
from tqdm import tqdm

def convert_raw_to_npy(source_dir, dest_dir, shape=(900, 1000), dtype=np.float32):
    os.makedirs(dest_dir, exist_ok=True)

    raw_files = [f for f in os.listdir(source_dir) if f.endswith(".raw")]

    print(f"Found {len(raw_files)} .raw files in {source_dir}")
    for filename in tqdm(raw_files, desc="Converting to .npy"):
        raw_path = os.path.join(source_dir, filename)
        npy_filename = filename.replace('.raw', '.npy')
        npy_path = os.path.join(dest_dir, npy_filename)

        try:
            arr = np.fromfile(raw_path, dtype=dtype).reshape(shape)
            np.save(npy_path, arr)
        except Exception as e:
            print(f"⚠️ Error converting {filename}: {e}")

    print(f"✅ Conversion complete. Saved to {dest_dir}")

# ----------- USAGE EXAMPLE ------------
if __name__ == "__main__":
    source_folder = r"E:\data\input\Sinogram"         # replace with your path
    destination_folder = r"E:\data_npy\input\Sinogram"      # replace with your path
    convert_raw_to_npy(source_folder, destination_folder)
# ----------- END OF USAGE EXAMPLE ------------