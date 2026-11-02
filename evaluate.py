import torch
import torch.nn.functional as F


def evaluate(model,device,loader):
    model.eval()

    total_loss = 0
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for images,labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = F.nll_loss(output,labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = output.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    print(f"平均损失:{avg_loss:.4f}")
    print(f"准确率:{accuracy*100:.2f}%")

    return avg_loss,accuracy
