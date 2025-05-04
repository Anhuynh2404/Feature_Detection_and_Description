import numpy as np

# class Conv2D:
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         self.stride = stride
#         self.padding = padding
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / (in_channels * kernel_size * kernel_size))
#         self.bias = np.zeros((out_channels, 1))

#     def forward(self, x):
#         x_padded = np.pad(x, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
#         batch_size, _, h, w = x.shape
#         out_h = (h + 2*self.padding - self.kernel_size) // self.stride + 1
#         out_w = (w + 2*self.padding - self.kernel_size) // self.stride + 1
#         out = np.zeros((batch_size, self.out_channels, out_h, out_w))
#         for b in range(batch_size):
#             for c_out in range(self.out_channels):
#                 for i in range(out_h):
#                     for j in range(out_w):
#                         h_start = i * self.stride
#                         h_end = h_start + self.kernel_size
#                         w_start = j * self.stride
#                         w_end = w_start + self.kernel_size
#                         region = x_padded[b, :, h_start:h_end, w_start:w_end]
#                         out[b, c_out, i, j] = np.sum(region * self.weights[c_out]) + self.bias[c_out]
#         return out

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], mode='constant')
    cols = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            cols[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return cols

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # He initialization
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        N, C, H, W = x.shape
        kH, kW = self.kernel_size, self.kernel_size
        out_h = (H + 2*self.padding - kH)//self.stride + 1
        out_w = (W + 2*self.padding - kW)//self.stride + 1

        col = im2col(x, kH, kW, self.stride, self.padding)  # shape: (N*out_h*out_w, C*kH*kW)
        col_W = self.weights.reshape(self.out_channels, -1).T  # shape: (C*kH*kW, out_channels)
        out = np.dot(col, col_W) + self.bias  # shape: (N*out_h*out_w, out_channels)

        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)  # (N, out_channels, out_h, out_w)
        return out


class BatchNorm2D:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=(0,2,3), keepdims=True)
            var = np.var(x, axis=(0,2,3), keepdims=True)
            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

class GlobalAvgPool:
    def forward(self, x):
        return np.mean(x, axis=(2,3), keepdims=True)

class Linear:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(out_features, in_features) * np.sqrt(2. / in_features)
        self.bias = np.zeros((out_features, 1))

    def forward(self, x):
        self.x = x  
        return np.dot(self.weights, x) + self.bias  
    
    def backward(self, grad_output):
        """
        grad_output: gradient từ loss w.r.t. output (shape: (out_features, batch_size))
        Trả về:
        - grad_input: gradient w.r.t. input x
        - grad_weights: gradient w.r.t. weights
        - grad_bias: gradient w.r.t. bias
        """
        batch_size = grad_output.shape[1]

        # Gradient của loss theo weights
        grad_weights = np.dot(grad_output, self.x.T) / batch_size  # shape: (out_features, in_features)

        # Gradient theo bias
        grad_bias = np.mean(grad_output, axis=1, keepdims=True)  # shape: (out_features, 1)

        # Gradient truyền về phía trước (cho lớp trước đó nếu cần)
        grad_input = np.dot(self.weights.T, grad_output)  # shape: (in_features, batch_size)

        return grad_input, grad_weights, grad_bias

