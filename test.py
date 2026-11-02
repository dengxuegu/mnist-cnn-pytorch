from pathlib import Path

import torch
from torchvision import datasets
from torch.utils.data import DataLoader

from model import Net
from data import build_transform
from evaluate import evaluate
from data import build_transform,DATA_DIR

def main():
    device = torch.device("cpu")

    model = Net().to(device)

    best_weights_path = Path(__file__).parent / "mnist_cnn_16ch_best.pt"
    state_dict = torch.load(
        best_weights_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    transform = build_transform()

    test_data = datasets.MNIST(
        root=str(Path(__file__).parent.parent / "data"),
        train=False,
        download=False,
        transform=transform,
    )

    test_loader = DataLoader(test_data, batch_size=1000,shuffle=False)

    print("最佳模型的测试集评估：")
    test_loss,test_accuracy = evaluate(model, device, test_loader)
    print("损失与准确率：",test_loss,test_accuracy)
    return test_loss,test_accuracy


if __name__== "__main__":
    main()