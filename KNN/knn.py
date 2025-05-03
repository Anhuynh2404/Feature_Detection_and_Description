import os
import cv2
import numpy as np
import time
from collections import Counter
import random
import matplotlib.pyplot as plt

class KNNImageClassifier:
    def __init__(self, k=5):
        """
        Khởi tạo bộ phân loại KNN
        
        Args:
            k (int): Số lượng láng giềng gần nhất
        """
        self.k = k
        self.X_train = None
        self.y_train = None
        self.class_names = None
    
    def _preprocess_image(self, image_path, target_size=(64, 64)):
        """
        Tiền xử lý ảnh: đọc, chuyển thành grayscale, thay đổi kích thước và làm phẳng
        
        Args:
            image_path (str): Đường dẫn đến file ảnh
            target_size (tuple): Kích thước mục tiêu (chiều rộng, chiều cao)
            
        Returns:
            np.ndarray: Vector đặc trưng đã được làm phẳng
        """
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            print(f"Không thể đọc ảnh: {image_path}")
            return None
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Thay đổi kích thước
        resized = cv2.resize(gray, target_size)
        
        # Chuẩn hóa
        normalized = resized / 255.0
        
        # Làm phẳng thành vector 1D
        flattened = normalized.flatten()
        
        return flattened
    
    def load_data(self, data_dir, train_ratio=0.8, random_state=42):
        """
        Tải dữ liệu từ thư mục và chia thành tập huấn luyện và kiểm tra
        
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa dữ liệu
            train_ratio (float): Tỉ lệ dữ liệu dùng cho huấn luyện
            random_state (int): Giá trị khởi tạo cho random
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, class_names)
        """
        features = []
        labels = []
        
        # Lấy tên các lớp (tên thư mục con)
        class_names = [d for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d))]
        
        print(f"Đã tìm thấy {len(class_names)} lớp: {class_names}")
        
        # Đọc dữ liệu từ từng lớp
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, class_name)
            print(f"Đang xử lý lớp: {class_name}")
            
            # Lấy danh sách các file ảnh trong thư mục
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"  - Tìm thấy {len(image_files)} ảnh")
            
            # Xử lý từng ảnh
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                img_features = self._preprocess_image(img_path)
                
                if img_features is not None:
                    features.append(img_features)
                    labels.append(class_idx)
        
        # Chuyển sang numpy array
        X = np.array(features)
        y = np.array(labels)
        
        # Trộn dữ liệu
        random.seed(random_state)
        indices = list(range(len(X)))
        random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Chia tập train/test
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Kích thước dữ liệu huấn luyện: {X_train.shape}")
        print(f"Kích thước dữ liệu kiểm tra: {X_test.shape}")
        
        # Lưu lại dữ liệu huấn luyện
        self.X_train = X_train
        self.y_train = y_train
        self.class_names = class_names
        
        return X_train, y_train, X_test, y_test, class_names
    
    def _euclidean_distance(self, x1, x2):
        """
        Tính khoảng cách Euclidean giữa hai vector
        
        Args:
            x1 (np.ndarray): Vector thứ nhất
            x2 (np.ndarray): Vector thứ hai
            
        Returns:
            float: Khoảng cách Euclidean
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _predict_single(self, x):
        """
        Dự đoán nhãn cho một mẫu duy nhất
        
        Args:
            x (np.ndarray): Vector đặc trưng
            
        Returns:
            int: Nhãn dự đoán
        """
        # Tính khoảng cách đến tất cả các điểm trong tập huấn luyện
        distances = []
        for idx, x_train in enumerate(self.X_train):
            dist = self._euclidean_distance(x, x_train)
            distances.append((dist, self.y_train[idx]))
        
        # Sắp xếp theo khoảng cách tăng dần
        distances.sort(key=lambda x: x[0])
        
        # Lấy k láng giềng gần nhất
        k_nearest = distances[:self.k]
        
        # Đếm số lần xuất hiện của mỗi nhãn
        k_nearest_labels = [label for _, label in k_nearest]
        most_common = Counter(k_nearest_labels).most_common(1)
        
        return most_common[0][0]
    
    def predict(self, X):
        """
        Dự đoán nhãn cho nhiều mẫu
        
        Args:
            X (np.ndarray): Ma trận các vector đặc trưng
            
        Returns:
            np.ndarray: Mảng các nhãn dự đoán
        """
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x))
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test):
        """
        Đánh giá mô hình trên tập kiểm tra
        
        Args:
            X_test (np.ndarray): Dữ liệu kiểm tra
            y_test (np.ndarray): Nhãn thực tế
            
        Returns:
            float: Độ chính xác
        """
        start_time = time.time()
        y_pred = self.predict(X_test)
        end_time = time.time()
        
        # Tính độ chính xác
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        
        print(f"Độ chính xác: {accuracy:.4f}")
        print(f"Thời gian dự đoán: {end_time - start_time:.2f} giây")
        
        # Tính confusion matrix
        conf_matrix = np.zeros((len(self.class_names), len(self.class_names)), dtype=int)
        for i in range(len(y_test)):
            conf_matrix[y_test[i]][y_pred[i]] += 1
        
        return accuracy, conf_matrix
    
    def predict_image(self, image_path):
        """
        Dự đoán nhãn cho một ảnh mới
        
        Args:
            image_path (str): Đường dẫn đến ảnh
            
        Returns:
            str: Tên lớp dự đoán
        """
        # Tiền xử lý ảnh
        features = self._preprocess_image(image_path)
        
        if features is None:
            return None
        
        # Dự đoán
        prediction = self._predict_single(features)
        
        return self.class_names[prediction]
    
    def visualize_confusion_matrix(self, conf_matrix):
        """
        Hiển thị confusion matrix
        
        Args:
            conf_matrix (np.ndarray): Ma trận nhầm lẫn
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Đánh dấu tọa độ
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)
        
        # Hiển thị giá trị
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, conf_matrix[i, j],
                         horizontalalignment="center",
                         color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('Nhãn thực tế')
        plt.xlabel('Nhãn dự đoán')
        plt.show()


# Hàm main để chạy thử nghiệm
def main():
    # Đường dẫn đến thư mục dữ liệu
    data_dir = "/home/an/an_workplace/Lab_CV/Data/PetImages"  # Thay đổi thành đường dẫn thực tế
    
    # Tạo bộ phân loại KNN với k=5
    classifier = KNNImageClassifier(k=5)
    
    # Tải và chia dữ liệu
    X_train, y_train, X_test, y_test, class_names = classifier.load_data(data_dir)
    
    # Kiểm tra mô hình
    print("\nĐánh giá mô hình:")
    accuracy, conf_matrix = classifier.evaluate(X_test, y_test)
    
    # Hiển thị confusion matrix
    classifier.visualize_confusion_matrix(conf_matrix)
    
    # Dự đoán một số ảnh mẫu từ tập kiểm tra
    print("\nDự đoán một số ảnh mẫu:")
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    
    for idx in sample_indices:
        true_label = class_names[y_test[idx]]
        predicted_label = class_names[classifier._predict_single(X_test[idx])]
        print(f"Ảnh {idx}: Nhãn thực tế: {true_label}, Dự đoán: {predicted_label}")
    
    # Thử nghiệm với các giá trị k khác nhau
    k_values = [1, 3, 5, 7, 9]
    accuracies = []
    
    print("\nThử nghiệm với các giá trị k khác nhau:")
    for k in k_values:
        classifier.k = k
        acc, _ = classifier.evaluate(X_test, y_test)
        accuracies.append(acc)
    
    # Vẽ biểu đồ độ chính xác theo k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Độ chính xác theo giá trị k')
    plt.xlabel('k')
    plt.ylabel('Độ chính xác')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()