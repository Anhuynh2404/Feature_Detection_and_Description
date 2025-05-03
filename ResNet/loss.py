import numpy as np

class CrossEntropyLoss:
    def forward(self, logits, labels):
        # logits: (batch_size, num_classes)
        # labels: (batch_size,)
        m = logits.shape[0]
        # Softmax
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        # Cross-entropy loss
        log_likelihood = -np.log(probs[range(m), labels])
        loss = np.sum(log_likelihood) / m
        return loss, probs

    def backward(self, probs, labels):
        # Gradient của cross-entropy w.r.t. logits
        m = probs.shape[0]
        grad = probs.copy()
        grad[range(m), labels] -= 1
        grad = grad / m
        return grad