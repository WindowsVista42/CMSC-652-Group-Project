import math
import pandas as pd
import skimage.metrics as skm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_test(device, model, test_loader, criterion):
    """
    Calculates the test loss and accuracy for the given model.

    Parameters:
    - model: Trained model for evaluation.
    
    Returns:
    - test_loss: Average test loss over the test dataset.
    - test_accuracy: Test accuracy over the test dataset.
    """
    model.eval()
    epoch_loss = 0
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_num, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            _, predicted = torch.max(output, 1)
            loss = criterion(output, labels)
            epoch_loss += loss.item()

            total_correct += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    # Average test loss
    avg_loss = epoch_loss / len(test_loader)
    avg_acc = total_correct / total_predictions
    
    return avg_loss, avg_acc

def get_predict(model, data_loader, device):
    """
    Get all predictions and true labels from the data loader.

    Parameters:
    - model: Trained model for predictions.
    - data_loader: DataLoader for the dataset.

    Returns:
    - y_true: List of true labels.
    - y_pred: List of predicted labels.
    """
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for batch_num, (images, labels) in enumerate(data_loader):
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    return true_labels, predictions

def remove_prefix(s):
    return s.replace("datasets/ctscan/raw\\", "")

def write_list_file(list_file, paths):
    if os.path.exists(list_file):
        os.remove(list_file)

    with open(list_file, "x") as f:
        for p in paths:
            f.write(f"{p.strip()}\n")

def load_list_file(list_file: str):
    with open(list_file, "r") as f:
        return list(map(lambda s: s.strip(), f.readlines()))

def eval_ssim(path_pair):
    ref = np.array(Image.open(path_pair[0]).convert("L"))
    watermarked = np.array(Image.open(path_pair[1]).convert("L"))
    return skm.structural_similarity(ref, watermarked, data_range=watermarked.max() - watermarked.min())

def eval_psnr(path_pair):
    ref = np.array(Image.open(path_pair[0]).convert("L"))
    watermarked = np.array(Image.open(path_pair[1]).convert("L"))
    return skm.peak_signal_noise_ratio(ref, watermarked, data_range=watermarked.max() - watermarked.min())

image_size = (224, 224)

train_transform = v2.Compose([
    v2.Resize(image_size),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(p = 0.5), # 50 % from images will apply to
    v2.RandomVerticalFlip(p = 0.5), # 50 % from images will apply to
    v2.RandomRotation(10),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = v2.Compose([
    v2.Resize(image_size),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform = {
    "train_transform":train_transform,
    "valid_transform":valid_transform
}

class CovidDataset(Dataset):
    def __init__(self, paths, transform = None, is_train =True):
        # data loadig
        self.paths = paths
        self.transform = transform
        self.is_train = is_train
    
    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        img = img.convert("RGB") # Some images 4 channels
        label = self.paths[index][-15:-10]
    
        if self.transform:
            if self.is_train:
                img = self.transform["train_transform"](img)
            else:
                img = self.transform["valid_transform"](img)
        
        return img, (1 if label == "Covid" else 0)
        
    def __len__(self):
        return len(self.paths)

class_names = ['COVID', 'non-COVID']

test_list_file = str("datasets/ctscan/test.txt")

watermarked_folders = [
    "datasets/ctscan/raw/",
    "datasets/ctscan/krawtchouk_0.500_0.500_50/",
    "datasets/ctscan/krawtchouk_0.500_0.500_100/",
    "datasets/ctscan/krawtchouk_0.500_0.500_200/",
    "datasets/ctscan/krawtchouk_0.500_0.500_300/",
    "datasets/ctscan/guo_zhuang_75/",
    "datasets/ctscan/guo_zhuang_300/",
    "datasets/ctscan/roni/",
]

df_names = [
    "raw",
    "krawtchouk p1=0.500 p2=0.500 strength=50",
    "krawtchouk p1=0.500 p2=0.500 strength=100",
    "krawtchouk p1=0.500 p2=0.500 strength=200",
    "krawtchouk p1=0.500 p2=0.500 strength=300",
    "guo_zhuang bits=75",
    "guo_zhuang bits=300",
    "roni",
]

nice_names = [
    "Unaltered",
    "Krawtchouk",
    "Krawtchouk",
    "Krawtchouk",
    "Krawtchouk",
    "Guo-Zhuang",
    "Guo-Zhuang",
    "RONI",
]

class ModelPerformanceReport:
    def __init__(self):
        self.test_list_file = str(test_list_file)
        self.test_rel_paths = load_list_file(str(test_list_file))
        self.test_raw = list(map(lambda s: f"datasets/ctscan/raw/{s}", self.test_rel_paths))
        self.watermarked_folders = watermarked_folders
        self.df_names = df_names
        self.nice_names = nice_names
        self.watermark_images_df = []
        self.df = pd.DataFrame()
        
        self.accs = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.true_pos = []
        self.false_neg = []
        self.false_pos = []
        self.true_neg  = []
        self.cms = []
        self.ssim_means = []
        self.ssim_stdevs = []
        self.psnr_means = []
        self.psnr_stdevs = []

    def evaluate(self, device, model, criterion):     
        print("Calculating image metrics...")
                
        def compute_stats(S, m, s):
            m.append(np.mean(S))
            s.append(np.std(S))

        for i, folder in enumerate(self.watermarked_folders):
            print(f"Calculating {folder}")
            test_watermarked = list(map(lambda s: f"{folder}{s}", self.test_rel_paths))
            test_path_pairs = list(zip(self.test_raw, test_watermarked))
            
            df = pd.DataFrame()
            
            df["rel path"] = self.test_rel_paths
            df["raw path"] = self.test_raw
            df["watermarked path"] = test_watermarked
            df["ssim"] = list(map(lambda x: eval_ssim(x), test_path_pairs))
            df["psnr"] = list(map(lambda x: eval_psnr(x), test_path_pairs))
        
            self.watermark_images_df.append(df)
            
            compute_stats(df["ssim"], self.ssim_means, self.ssim_stdevs)
            compute_stats(df["psnr"], self.psnr_means, self.psnr_stdevs)
            print("Avg SSIM:", self.ssim_means[i])
            print("Avg PSNR (dB):", self.psnr_means[i])
        
        print("Finished calculating image metrics.")
        print()

        print("Calculating performance metrics...")
        
        for (df, df_name) in zip(self.watermark_images_df, self.df_names):
            print(f"Testing {df_name}")
            test_data = CovidDataset(df["watermarked path"], transform, is_train=False)
            test_loader = DataLoader(test_data, shuffle=False, batch_size=16)
            
            test_loss, test_accuracy = get_test(device, model, test_loader, criterion)
            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%')
        
            self.accs.append(test_accuracy)
        
            y_true, y_pred = get_predict(model, test_loader, device)
            report = classification_report(y_true, y_pred, target_names= class_names)
            print(report)
        
            precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred, labels=class_names, average='binary')
        
            self.precisions.append(precision)
            self.recalls.append(recall)
            self.f1_scores.append(f1_score)
        
            cm = confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(cm, class_names=class_names, figsize=(8,6))
            plt.title(f"Confusion Matrix: {df_name}")
            #plt.savefig(f"mobilenetv2_confusion_matrix_{df_name}.png")
            plt.show()
        
            self.true_pos.append(cm[0][0])
            self.false_neg.append(cm[0][1])
            self.false_pos.append(cm[1][0])
            self.true_neg.append(cm[1][1])
            self.cms.append(cm)
        
        df = pd.DataFrame()
        df["Name"] = self.nice_names
        df["Strength"] = ("N/A", 50, 100, 200, 300, "N/A", "N/A", "N/A")
        df["Position"] = ("N/A", (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), "N/A", "N/A", "N/A")
        df["L-Bits"] = ("N/A", 1024, 1024, 1024, 1024, 75, 300, 1024)
        df["Mean SSIM"] = self.ssim_means
        df["Stdev SSIM"] = self.ssim_stdevs
        df["Mean PSNR (dB)"] = self.psnr_means
        df["Stdev PSNR (dB)"] = self.psnr_stdevs
        df["Accuracy"] = self.accs
        df["Precision"] = self.precisions
        df["Recall"] = self.recalls
        df["F1 Score"] = self.f1_scores
        df["Number of Test Images"] = [len(self.test_rel_paths)] * len(self.watermark_images_df)
        df["True Positives"] = self.true_pos
        df["False Negatives"] = self.false_neg
        df["True Negatives"] = self.true_neg
        df["False Positives"] = self.false_pos

        # Nice output for unaltered
        if df["Mean PSNR (dB)"][0] == math.inf:
            df["Mean PSNR (dB)"][0] = "∞"
        if math.isnan(df["Stdev PSNR (dB)"][0]):
            df["Stdev PSNR (dB)"][0] = "Undefined"
        
        # df.to_csv("mobilenetv2_results.csv")
        self.df = df
        
        print("Finished calculating performance metrics.")