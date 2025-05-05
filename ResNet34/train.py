from data import *
from model import *
from test import test
import matplotlib.pyplot as plt

class trainer:
    def __init__(self, model, dataset, num_classes, init_lr):
        self.dataset = dataset
        self.net = model
        self.lr = init_lr
        self.cls_num = num_classes

    def set_lr(self, lr):
        self.lr = lr

    def iterate(self):
        images, labels = self.dataset.get_next_batch()

        out_tensor = self.net.forward(images)

        if self.cls_num > 1:
            one_hot_labels = np.eye(self.cls_num)[labels.reshape(-1)].reshape(out_tensor.shape)
        else:
            one_hot_labels = labels.reshape(out_tensor.shape)
            
        # loss = np.sum(-one_hot_labels * np.log(out_tensor)-(1-one_hot_labels) * np.log(1 - out_tensor)) / self.dataset.batch_size
        # loss = -np.sum(one_hot_labels * np.log(out_tensor + 1e-9)) / self.dataset.batch_size
        # # out_diff_tensor = (out_tensor - one_hot_labels) / out_tensor / (1 - out_tensor) / self.dataset.batch_size
        # out_diff_tensor = (out_tensor - one_hot_labels) / self.dataset.batch_size

        labels = labels.reshape(-1, 1).astype(np.float32)
        loss = -np.mean(labels * np.log(out_tensor + 1e-9) + (1 - labels) * np.log(1 - out_tensor + 1e-9))
        out_diff_tensor = (out_tensor - labels)  

        
        self.net.backward(out_diff_tensor, self.lr)
        
        return loss



if __name__ == '__main__':
    batch_size = 4
    image_h = 128
    image_w = 128
    num_classes = 2
    dataset = dataloader("/home/an/an_workplace/Lab_CV/train.txt", batch_size, image_h, image_w)

    model = resnet34(num_classes)

    init_lr = 0.01
    train = trainer(model, dataset, num_classes, init_lr)
    loss = []
    accurate = []
    temp = 0

    model.train()
    plt.figure(figsize=(10,5))
    plt.ion()
    images, labels = dataset.get_next_batch()
    print("Image shape:", images.shape)
    print("Labels:", labels[:2])

    # out = model.forward(images)
    # print("Out shape:", out.shape)
    # print("Out sample:", out[0])
    # print("Out sum:", out[0].sum())  # ~1 nếu softmax đúng
    # loss = train.iterate()
    # print("Loss:", loss)

    for i in range(1000):
        temp += train.iterate()
        if i % 10 == 0 and i != 0:
            loss.append(temp / 10)
            print("iteration = {} || loss = {}".format(str(i), str(temp/10)))
            temp = 0
            if i % 100 == 0:
                model.eval()
                accurate.append(test(model, "/home/an/an_workplace/Lab_CV/test.txt", image_h, image_w))
                model.save("model2")
                model.train()

        if i % 1000 == 0:
            plt.cla()
            plt.subplot(1,2,1)
            plt.plot(loss)
            plt.title("Loss")
            plt.subplot(1,2,2)
            plt.plot(accurate)
            plt.title("Accuracy")
            plt.show()


        if i == 200:
            train.set_lr(0.001)

    plt.ioff()
    plt.show()