class SGD:
    def __init__(self, parameters, lr=0.01):
        # parameters: danh sách các tuple (param_tensor, grad_tensor)
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param, grad in self.parameters:
            param -= self.lr * grad

    def zero_grad(self):
        # không dùng trong dự án này vì không lưu gradient trực tiếp trong object param
        pass