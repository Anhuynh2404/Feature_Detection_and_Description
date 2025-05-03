import os
import numpy as np
from PIL import Image

class AnimalDataset:
    def __init__(self, root_dir, image_size=(224, 224)):
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.image_size)
        img = np.asarray(img) / 255.0  # Normalize to [0, 1]
        img = img.transpose((2, 0, 1))  # Convert to CHW format
        return img, label

    def get_batches(self, batch_size=32, shuffle=True):
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(self), batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            images, labels = zip(*[self[i] for i in batch_idx])
            yield np.array(images), np.array(labels)