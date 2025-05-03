import os
import cv2
import numpy as np
import random

class DataLoader:
    def __init__(self, image_size=(64, 64), grayscale=True):
        self.image_size = image_size
        self.grayscale = grayscale
        self.X_train = None
        self.y_train = None
        self.class_names = None

    def _preprocess_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[Read Image Error] {img_path}")
                return None
            
            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_resized = cv2.resize(img, self.image_size)
            img_normalized = img_resized / 255.0

            # Chuyển về 1D
            return img_normalized.flatten()
        except Exception as e:
            print(f"[Processor Image Error] {img_path}: {e}")
            return None

    def load_data(self, data_dir, train_ratio=0.8, random_state=42):
        """
        Load và chia dữ liệu thành tập huấn luyện / kiểm tra
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test, class_names)
        """
        features, labels = [], []
        
        class_names = [d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))]

        class_names.sort()  # Đảm bảo nhãn luôn nhất quán
        print(f"Find {len(class_names)} Class: {class_names}")

        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, class_name)
            print(f"Processing  '{class_name}'...")

            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            print(f"  - Find {len(image_files)} Images")

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                img_features = self._preprocess_image(img_path)

                if img_features is not None:
                    features.append(img_features)
                    labels.append(class_idx)

        X = np.array(features)
        y = np.array(labels)

        # Shuffle
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]

        # Split
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Train Data: {X_train.shape}")
        print(f"Test Data:   {X_test.shape}")

        self.X_train = X_train
        self.y_train = y_train
        self.class_names = class_names

        return X_train, y_train, X_test, y_test, class_names
