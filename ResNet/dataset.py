import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class AnimalDataset:
    def __init__(self, root_dir, image_size=(128, 128)):
        self.root_dir = root_dir
        self.image_size = image_size
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_folder):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_folder, img_name))
                    self.labels.append(self.class_to_idx[cls])

        self.mean, self.std = self.compute_mean_std()

    def compute_mean_std(self):
        print("Tính mean và std trên toàn bộ tập ảnh...")
        pixel_sum = 0
        pixel_sq_sum = 0
        num_pixels = 0

        for path in self.image_paths:
            img = Image.open(path).convert('RGB')
            img = img.resize(self.image_size)
            arr = np.asarray(img) / 255.0  # scale về [0, 1]
            pixel_sum += arr.sum(axis=(0, 1))
            pixel_sq_sum += (arr ** 2).sum(axis=(0, 1))
            num_pixels += arr.shape[0] * arr.shape[1]

        mean = pixel_sum / num_pixels        # shape: (3,)
        std = np.sqrt(pixel_sq_sum / num_pixels - mean ** 2)
        print(f"Mean: {mean}, Std: {std}")
        return mean.reshape((3, 1, 1)), std.reshape((3, 1, 1))  # reshape để broadcast

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.image_size)
        img = np.asarray(img) / 255.0  # Normalize to [0, 1]
        img = img.transpose((2, 0, 1))  # Convert to CHW format

        img = (img - self.mean) / self.std
        return img, label

    def get_batches(self, batch_size=32, shuffle=True):
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(self), batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            images, labels = zip(*[self[i] for i in batch_idx])
            yield np.array(images), np.array(labels)

    
    def plot_dataset_statistics(dataset):
        means = []
        stds = []

        print("📊 Đang tính mean/std từng ảnh để vẽ histogram...")
        for path in dataset.image_paths:
            img = Image.open(path).convert('RGB').resize(dataset.image_size)
            arr = np.asarray(img) / 255.0  # [H, W, 3]
            arr = arr.reshape(-1, 3)       # [H*W, 3]
            means.append(np.mean(arr, axis=0))  # mean RGB
            stds.append(np.std(arr, axis=0))    # std RGB

        means = np.array(means)
        stds = np.array(stds)

        # Vẽ biểu đồ
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            axes[0].hist(means[:, i], bins=30, color=color, alpha=0.6, label=f'{color.upper()}')
            axes[1].hist(stds[:, i], bins=30, color=color, alpha=0.6, label=f'{color.upper()}')

        axes[0].set_title('Phân phối Mean theo kênh màu')
        axes[1].set_title('Phân phối Std theo kênh màu')
        axes[0].set_xlabel('Giá trị mean')
        axes[1].set_xlabel('Giá trị std')
        axes[0].legend()
        axes[1].legend()
        plt.show()