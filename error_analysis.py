from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision import datasets

from model import Net
from data import DATA_DIR, build_transform
from collections import Counter

def collect_errors(model, loader, device):
    model.eval()
    errors = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            predictions = output.argmax(dim=1)

            # 找到预测错误的样本
            wrong_mask = predictions != labels
            wrong_labels = labels[wrong_mask]
            wrong_predictions = predictions[wrong_mask]

            # 将真实类别和预测类别加入 errors
            for true_label, predicted_label in zip(wrong_labels, wrong_predictions):
                errors.append({
                    "true_label": true_label.item(),
                    "predicted_label": predicted_label.item(),
                })

    return errors

def main():
    # 1. 创建 CPU 模型并加载最佳权重
    device = torch.device("cpu")
    model = Net().to(device)

    weights_path = (
            Path(__file__).resolve().parent/ "mnist_cnn_16ch_best.pt"
    )

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"未找到权重文件：{weights_path}"
        )

    state_dict = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    # 2. 用相同种子重建原来的验证集
    dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=build_transform(),
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [54000, 6000],
        generator=torch.Generator().manual_seed(1),
    )
    # 3. 创建验证集 DataLoader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1000,
        shuffle=False
    )
    # 4. 调用 collect_errors()
    errors = collect_errors(model, val_loader, device)
    print(f"验证集预测错误总数：{len(errors)}")
    # 5. 打印错误总数和前 10 条记录

    pair_counts = Counter()

    for error in errors:
        pair = (error["true_label"], error["predicted_label"])
        pair_counts[pair] += 1

    print("出现次数最多的 5 种误分类：")
    for (true_label, predicted_label), count in pair_counts.most_common(5):
        print(f"{true_label} → {predicted_label}：{count} 次")


if __name__ == "__main__":
    main()