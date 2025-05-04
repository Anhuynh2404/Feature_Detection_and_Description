# ResNet-from-Scratch for Animal Classification (NumPy Only)

This project implements a ResNet-34 model **from scratch** using **NumPy and OpenCV only**, for classifying animal images into 10 classes.

> No PyTorch. No TensorFlow. 100% manually-coded deep learning.

---

## 🐾 Dataset Format

Place your animal image dataset into this folder structure:

```
/home/an/an_workplace/Lab_CV/Data/animals/
├── cat/
├── dog/
├── horse/
├── elephant/
├── ... (total 10 folders)
```

Each folder should contain images of that animal class.

---

## 🛠 Setup

Make sure you have Python 3 and the following libraries:

```bash
pip install numpy opencv-python matplotlib
```

---

## 📁 File Structure

```
.
├── components.py       # Conv, BN, ReLU, FC, etc. written in NumPy
├── model.py            # ResNet34 forward/backward from scratch
├── data.py             # Dataloader for reading and batching
├── train.py            # Training loop
├── test.py             # Evaluation script
├── demo_test.py        # Top-3 inference script
├── make_dataset_list.py # Generate train.txt / test.txt from folder structure
```

---

## 🔧 Step-by-step Usage

### 1. Generate Dataset Lists

```bash
python make_dataset_list.py
```

This will generate `train.txt` and `test.txt` with format:

```
/full/path/to/image.jpg 3
/full/path/to/image.jpg 7
...
```

### 2. Train the Model

```bash
python train.py
```

* Resize: 128x128
* Batch size: 8
* Classes: 10
* Save every 100 iterations to `model2/`

### 3. Evaluate Accuracy

```bash
# Automatically runs every 100 iterations inside train.py
```

Or manually:

```python
from model import resnet34
from test import test

net = resnet34(10)
net.load("model2")
acc = test(net, "test.txt", 128, 128)
print("Accuracy:", acc)
```

### 4. Top-3 Demo

Prepare a text file with:

```
filename.jpg 3
```

Then run:

```bash
python demo_test.py
```

---

## 🧠 Notes

* This implementation uses sigmoid + binary cross-entropy with one-hot labels.
* Label values must be from 0 to 9.
* Resize is performed manually using OpenCV.
* No data augmentation is used (can be added).

---

## ✨ Credits

This project is a customized fork and extension of a minimal NumPy ResNet implementation, tailored to real-world datasets like animal classification.

Enjoy building deep learning from the ground up! 💪🐍
