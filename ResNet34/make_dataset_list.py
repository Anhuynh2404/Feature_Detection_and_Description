import os
import random

def generate_dataset_lists(data_dir, output_dir="./", image_exts=('.jpg', '.jpeg', '.png'), train_ratio=0.8):
    image_paths = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(image_exts):
                full_path = os.path.abspath(os.path.join(class_dir, fname))
                image_paths.append((full_path, class_to_idx[class_name]))

    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * train_ratio)
    train_list = image_paths[:split_idx]
    test_list = image_paths[split_idx:]

    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        for path, label in train_list:
            f.write(f"{path} {label}\n")

    with open(os.path.join(output_dir, "test.txt"), "w") as f:
        for path, label in test_list:
            f.write(f"{path} {label}\n")

    print("Train samples:", len(train_list))
    print("Test samples :", len(test_list))
    print("Class map    :", class_to_idx)

if __name__ == "__main__":
    data_dir = "/home/an/an_workplace/Lab_CV/Data/animals"
    generate_dataset_lists(data_dir)
