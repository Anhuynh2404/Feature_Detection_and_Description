import numpy as np
import time

# def train(model, dataset, loss_fn, optimizer, epochs=5, batch_size=8):
#     losses = []
#     for epoch in range(epochs):
#         total_loss = 0
#         for X_batch, y_batch in dataset.get_batches(batch_size):
#             logits = model.forward(X_batch)
#             loss, probs = loss_fn.forward(logits, y_batch)
#             total_loss += loss

#             dlogits = loss_fn.backward(probs, y_batch)

#             optimizer.step() 

#         avg_loss = total_loss / (len(dataset) // batch_size)
#         losses.append(avg_loss)
#         print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}", flush=True)

# def train(model, dataset, loss_fn, optimizer, epochs=1, batch_size=1):
#     import time
#     losses = []
#     for epoch in range(epochs):
#         total_loss = 0
#         start_time = time.time()
#         for i, (X_batch, y_batch) in enumerate(dataset.get_batches(batch_size)):
#             logits = model.forward(X_batch)
#             loss, probs = loss_fn.forward(logits, y_batch)
#             dlogits = loss_fn.backward(probs, y_batch)
#             optimizer.step()
#             total_loss += loss

#             if i % 1 == 0:
#                 print(f"  Batch {i+1}: Loss = {loss:.4f}", flush=True)

#         avg_loss = total_loss / (len(dataset) // batch_size)
#         losses.append(avg_loss)
#         print(f"Epoch {epoch+1} done in {time.time() - start_time:.2f}s, Avg Loss: {avg_loss:.4f}", flush=True)
#     return losses
def save_model(model, path="checkpoint.npz"):
    np.savez(path, weights=model.fc.weights, bias=model.fc.bias)

def train(model, dataset, loss_fn, optimizer, epochs=5, batch_size=4):
    losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} bắt đầu...\n")
        total_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(dataset.get_batches(batch_size)):
            print(f"Batch {batch_idx+1} ─ Bắt đầu")

            start = time.time()
            logits = model.forward(X_batch)
            

            print(" [Model] Forward")
            logits = model.forward(X_batch)
            print("[Model] Forward DONE in", time.time() - start, "seconds")

            print(" [Loss] Tính toán CrossEntropy")
            loss, probs = loss_fn.forward(logits, y_batch)
            total_loss += loss

            print(" [Loss] Backward từ output")
            dlogits = loss_fn.backward(probs, y_batch).T  # shape: (num_classes, batch)

            print(" [Linear] Backward FC layer")
            grad_input, grad_w, grad_b = model.fc.backward(dlogits)

            print(" [Optimizer] Cập nhật tham số")
            optimizer.parameters = [
                (model.fc.weights, grad_w),
                (model.fc.bias, grad_b)
            ]
            optimizer.step()

            print("  Batch done\n")

        avg_loss = total_loss / (len(dataset) // batch_size)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} hoàn thành - Loss TB: {avg_loss:.4f}")
        save_model(model, f"checkpoint_epoch{epoch+1}.npz")

    return losses


def predict(model, X):
    logits = model.forward(X)
    return np.argmax(logits, axis=1)

def evaluate(model, dataset, batch_size=8):
    correct = 0
    total = 0
    for X_batch, y_batch in dataset.get_batches(batch_size, shuffle=False):
        preds = predict(model, X_batch)
        correct += np.sum(preds == y_batch)
        total += len(y_batch)
    acc = correct / total * 100
    print(f"Accuracy: {acc:.2f}%")
    return acc
