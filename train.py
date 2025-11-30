import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import torch
from ultralytics import YOLO
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============= DATA PROCESSOR =============
class DataProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.classes = []
        
    def analyze_dataset(self):
        """Analyze and count images per class"""
        plant_village = self.input_path if any(d.is_dir() for d in self.input_path.iterdir()) else self.input_path / "PlantVillage"
        self.classes = sorted([d.name for d in plant_village.iterdir() if d.is_dir()])
        self.plant_village_path = plant_village
        
        class_counts = {}
        for cls in self.classes:
            count = len([f for f in (plant_village / cls).iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            class_counts[cls] = count
            print(f"  {cls}: {count} images")
        return class_counts
    
    def create_dataset(self, test_size=0.2, val_size=0.1):
        """Create train/val/test splits"""
        dataset_path = self.output_path / "classification"
        
        for split in ['train', 'val', 'test']:
            for cls in self.classes:
                (dataset_path / split / cls).mkdir(parents=True, exist_ok=True)
        
        for cls in tqdm(self.classes, desc="Processing"):
            cls_path = self.plant_village_path / cls
            images = [f for f in cls_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            train_imgs, temp = train_test_split(images, test_size=test_size+val_size, random_state=42)
            val_imgs, test_imgs = train_test_split(temp, test_size=test_size/(test_size+val_size), random_state=42)
            
            for img_list, split in [(train_imgs, 'train'), (val_imgs, 'val'), (test_imgs, 'test')]:
                for img in img_list:
                    shutil.copy2(img, dataset_path / split / cls / img.name)
        
        with open(dataset_path / 'data.yaml', 'w') as f:
            yaml.dump({'path': str(dataset_path.absolute()), 'train': 'train', 'val': 'val', 
                      'test': 'test', 'nc': len(self.classes), 'names': self.classes}, f)
        
        return dataset_path

# ============= MODEL =============
class PlantDiseaseModel:
    def __init__(self):
        self.model = None
        self.classes = []
        
    def train(self, data_path, epochs=10, img_size=224, batch_size=16):
        """Train YOLOv8 classification model"""
        print("🚀 Training model...")
        self.model = YOLO('yolov8n-cls.pt')
        
        with open(data_path / 'data.yaml', 'r') as f:
            self.classes = yaml.safe_load(f)['names']
        
        results = self.model.train(
            data=str(data_path), epochs=epochs, imgsz=img_size, batch=batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu', save=True,
            patience=20, project='plant_disease', name='exp'
        )
        print("✅ Training completed!")
        return results
    
    def predict(self, image_path, conf=0.5):
        """Predict single image"""
        return self.model(image_path, conf=conf)
    
    def evaluate(self, test_path):
        """Generate confusion matrix data"""
        true_labels, pred_labels = [], []
        
        for class_dir in test_path.iterdir():
            if class_dir.is_dir():
                true_label = self.classes.index(class_dir.name)
                for img in class_dir.iterdir():
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        pred_label = self.model(str(img), verbose=False)[0].probs.top1
                        true_labels.append(true_label)
                        pred_labels.append(pred_label)
        
        return true_labels, pred_labels

# ============= VISUALIZER =============
class Visualizer:
    @staticmethod
    def plot_distribution(class_counts):
        """Plot class distribution"""
        plt.figure(figsize=(15, 8))
        plt.bar(class_counts.keys(), class_counts.values(), color='steelblue', edgecolor='black')
        plt.title('Class Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Disease Classes')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_predictions(results, max_imgs=30):
        """Visualize predictions"""
        n = min(len(results), max_imgs)
        rows = (n + 3) // 4
        cols = min(4, n)
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        axes = axes.flatten() if n > 1 else [axes]
        
        for i, result in enumerate(results[:n]):
            img = cv2.cvtColor(cv2.imread(result.path), cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].axis('off')
            
            if hasattr(result, 'probs'):
                cls_name = result.names[result.probs.top1]
                conf = result.probs.top1conf.item()
                axes[i].set_title(f'{cls_name}\nConf: {conf:.3f}', fontweight='bold')
        
        for j in range(n, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(true_labels, pred_labels, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        print("\n" + classification_report(true_labels, pred_labels, target_names=class_names))
        return cm

# ============= MAIN PIPELINE =============
def train_and_evaluate(input_path, output_path="./output", epochs=10, batch_size=16, images_per_class=3):
    """Complete training and evaluation pipeline"""
    
    print("=" * 60)
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    print(f"🔢 Epochs: {epochs} | Batch: {batch_size} | Viz Images/Class: {images_per_class}")
    print("=" * 60)
    
    # 1. Process Dataset
    processor = DataProcessor(input_path, output_path)
    class_counts = processor.analyze_dataset()
    dataset_path = processor.create_dataset()
    
    # 2. Visualize Distribution
    print("\n📊 Visualizing class distribution...")
    Visualizer.plot_distribution(class_counts)
    
    # 3. Train Model
    model = PlantDiseaseModel()
    model.train(dataset_path, epochs=epochs, batch_size=batch_size)
    
    # 4. Confusion Matrix
    print("\n📊 Generating confusion matrix...")
    true_labels, pred_labels = model.evaluate(dataset_path / 'test')
    Visualizer.plot_confusion_matrix(true_labels, pred_labels, model.classes)
    
    # 5. Visualize Predictions from Multiple Classes
    print(f"\n📊 Visualizing predictions from multiple classes...")
    test_path = dataset_path / 'test'
    sample_images = []
    
    for class_dir in sorted(test_path.iterdir()):
        if class_dir.is_dir():
            class_images = [f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            sample_images.extend(class_images[:images_per_class])
            if len(sample_images) >= 30:
                break
    
    predictions = []
    for img in tqdm(sample_images, desc="Predicting"):
        predictions.extend(model.predict(str(img)))
    
    Visualizer.plot_predictions(predictions)
    
    print("\n🎉 Complete! Model saved in: Plant_disease/exp/weights/best.pt")
    return model

# ============= INFERENCE =============
def predict_image(model_path, image_path):
    """Predict single image"""
    model = PlantDiseaseModel()
    model.model = YOLO(model_path)
    results = model.predict(image_path)
    
    for r in results:
        cls_name = r.names[r.probs.top1]
        conf = r.probs.top1conf.item()
        print(f"Predicted: {cls_name} (Confidence: {conf:.3f})")
    return results

# ============= EXECUTION =============
if __name__ == "__main__":
    
    # CONFIGURATION
    INPUT_PATH = "PlantVillage"
    OUTPUT_PATH = "./plant_disease_output"
    EPOCHS = 10
    BATCH_SIZE = 16
    IMAGES_PER_CLASS = 3 
    
    # RUN
    model = train_and_evaluate(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        images_per_class=IMAGES_PER_CLASS
    )
    
    # INFERENCE EXAMPLE (uncomment to use)
    # predict_image("plant_disease/exp/weights/best.pt", "test_image.jpg")
