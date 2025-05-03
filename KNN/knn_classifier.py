import numpy as np
from collections import Counter
import time
from distance import DistanceCalculator
from DataLoader import DataLoader

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.distance_calculator = DistanceCalculator()

    def fit(self, X_train, y_train, class_names):
        self.X_train = X_train
        self.y_train = y_train
        self.class_names = class_names

    def _calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return self.distance_calculator.euclidean(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self.distance_calculator.manhattan(x1, x2)
        elif self.distance_metric == 'minkowski':
            return self.distance_calculator.minkowski(x1, x2)
        else:
            raise ValueError("Unknown distance metric")

    def _predict_single(self, x):
        distances = [(self._calculate_distance(x, x_train), self.y_train[i])
                     for i, x_train in enumerate(self.X_train)]
        distances.sort(key=lambda x: x[0])
        k_labels = [label for _, label in distances[:self.k]]
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def evaluate(self, X_test, y_test):
        start = time.time()
        y_pred = self.predict(X_test)
        end = time.time()
        acc = np.mean(y_pred == y_test)
        print("\n=== Evaluate Model ===")
        print(f"Number of test sample: {len(y_test)}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Predict time: {end - start:.2f} giây")

        # Confusion matrix
        n_classes = len(self.class_names)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(len(y_test)):
            cm[y_test[i]][y_pred[i]] += 1
        return acc, cm

    def predict_image(self, image_path, preprocess_func):
        features = preprocess_func(image_path)
        if features is None:
            return None
        label_idx = self._predict_single(features)
        return self.class_names[label_idx]
