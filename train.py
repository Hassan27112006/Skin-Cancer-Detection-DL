import torch
import numpy as np
from sklearn.metrics import accuracy_score

def train_model(model, train_loader, test_loader, epochs, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_acc_list, test_acc_list = [], []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_acc_list.append(train_acc)

        test_acc, _, _, _ = evaluate(model, test_loader, device)
        test_acc_list.append(test_acc)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f} Test Acc={test_acc:.4f}")

    return train_acc_list, test_acc_list


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, 1)

            _, predicted = outputs.max(1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    return acc, np.array(y_true), np.array(y_pred), np.array(y_prob)
