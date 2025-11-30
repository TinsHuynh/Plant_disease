import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO
import os

class PlantDiseaseApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Plant Disease Detection")
        self.master.geometry("800x600")
        
        # Tải mô hình đã huấn luyện
        self.model = YOLO("plant_disease/exp/weights/best.pt")  # Đảm bảo đường dẫn đúng với mô hình của bạn

        # Tiêu đề
        self.title_label = tk.Label(self.master, text="Plant Disease Detection", font=("Arial", 24))
        self.title_label.pack(pady=20)

        # Nút tải ảnh
        self.upload_button = tk.Button(self.master, text="Upload Image", command=self.upload_image, font=("Arial", 14))
        self.upload_button.pack(pady=20)

        # Hiển thị hình ảnh tải lên
        self.image_label = tk.Label(self.master)
        self.image_label.pack(pady=20)

        # Hiển thị kết quả dự đoán
        self.result_label = tk.Label(self.master, text="Prediction Result", font=("Arial", 16))
        self.result_label.pack(pady=20)
        
    def upload_image(self):
        """Hàm tải hình ảnh và dự đoán kết quả"""
        # Mở cửa sổ chọn file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        
        if file_path:
            # Hiển thị hình ảnh tải lên
            img = Image.open(file_path)
            img = img.resize((300, 300))  # Resize để vừa với giao diện
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk  # Cần giữ tham chiếu đến ảnh để không bị mất

            # Dự đoán với mô hình YOLO
            self.predict_image(file_path)

    def predict_image(self, image_path):
        """Dự đoán bệnh từ ảnh"""
        results = self.model(image_path)  # Dự đoán từ mô hình YOLO
        
        # Kiểm tra kết quả dự đoán
        if results:
            # Lấy nhãn và độ tin cậy (confidence)
            pred_class = results[0].names[results[0].probs.top1]
            conf = results[0].probs.top1conf.item()
            
            # Hiển thị kết quả
            self.result_label.config(text=f"Predicted: {pred_class} (Confidence: {conf:.3f})")
        else:
            self.result_label.config(text="No disease detected or invalid image.")


# Chạy ứng dụng Tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseApp(root)
    root.mainloop()
