import kagglehub
import os
import shutil
import glob

XRAY_RAW_PATH = "datasets/ct-scan/raw"
MRI_RAW_PATH = "datasets/ct-scan/raw"
CTSCAN_RAW_PATH = "datasets/ct-scan/raw"

def download_dataset(kaggle_path, output_folder):
    print("Downloading dataset:", kaggle_path)

    path = kagglehub.dataset_download(kaggle_path)
    print("Dataset downloaded to:", path)

    # move files
    items = glob.glob(os.path.join(path, "*"))
    for item in items:
        move_path = os.path.join(output_folder, os.path.basename(item))

        # "Fresh install" of the dataset
        if os.path.exists(move_path):
            shutil.rmtree(move_path)

        shutil.move(item, output_folder)
    
    # post-cleanup folder so kaggle plays nice
    shutil.rmtree(path)

    print("Moved files to:", output_folder)


if __name__ == "__main__":
    download_dataset("sachinkumar413/cxr-2-classes", XRAY_RAW_PATH)
    print() # makes the output a little nicer
    download_dataset("masoudnickparvar/brain-tumor-mri-dataset", MRI_RAW_PATH)
    print() # makes the output a little nicer
    download_dataset("plameneduardo/sarscov2-ctscan-dataset", CTSCAN_RAW_PATH)