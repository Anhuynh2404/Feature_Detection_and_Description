import cv2
import numpy as np

def get_test_images(filename, image_h, image_w):
    with open(filename) as file:
        filelist = file.readlines()
    img_tensor = np.zeros([len(filelist), 3, image_h, image_w])
    label_tensor = np.zeros([len(filelist)])
    for i in range(len(filelist)):
        path, label = filelist[i].split(" ")
        img = cv2.imread(path)
        img = cv2.resize(img, (image_w, image_h))
        img = img[:,:,::-1].astype(np.float32)
        img_tensor[i] = img.transpose(2,0,1)
        label_tensor[i] = int(label)
    return img_tensor, label_tensor

def test(net, filename, image_h=128, image_w=128):
    net.eval()
    images, labels = get_test_images(filename, image_h, image_w)
    infers = np.zeros([images.shape[0]],dtype=np.int32)

    print(f"Testing {images.shape[0]} image...")
    for i in range(images.shape[0]):
        if i % 100 == 0:
            print(f"→ Inference ảnh {i}/{images.shape[0]}")
        infers[i] = net.inference(images[i].reshape(1,3,image_h, image_w))

    acc = np.sum(infers == labels) / infers.shape[0]
    print(f"Accuracy: {acc}")
    return acc 