import kagglehub
import os
import shutil
import glob

# XRAY_RAW_PATH = "datasets/xray/raw"
# MRI_RAW_PATH = "datasets/mri/raw"
CTSCAN_RAW_PATH = "datasets/ctscan/raw"

# Download dataset from kaggle and move it to the output folder.
def download_dataset(kaggle_path, output_folder):
    print("Downloading dataset:", kaggle_path)

    path = kagglehub.dataset_download(kaggle_path)
    print("Dataset downloaded to:", path)

    # Recreate folder structure if it doesn't exist for some reason.
    os.makedirs(output_folder, exist_ok=True)

    # Move files.
    items = glob.glob(os.path.join(path, "*"))
    for item in items:
        move_path = os.path.join(output_folder, os.path.basename(item))

        # "Fresh install" of the dataset.
        if os.path.exists(move_path):
            shutil.rmtree(move_path)

        shutil.move(item, output_folder)
    
    # Post-cleanup folder so kaggle plays nice.
    shutil.rmtree(path)

    print("Moved files to:", output_folder)


if __name__ == "__main__":
    # download_dataset("sachinkumar413/cxr-2-classes", XRAY_RAW_PATH)
    # print() # Nicer output.
    # download_dataset("masoudnickparvar/brain-tumor-mri-dataset", MRI_RAW_PATH)
    # print() # Nicer output.
    download_dataset("plameneduardo/sarscov2-ctscan-dataset", CTSCAN_RAW_PATH)