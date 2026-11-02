from pathlib import Path
import argparse
import torch
from model import Net
from torchvision import datasets, transforms
from data import build_transform
from data import build_transform,DATA_DIR

model = Net()

weights_path = Path(__file__).parent / "mnist_cnn_16ch_best.pt"
print("加载权重：", weights_path.name)

if not weights_path.is_file():
    raise FileNotFoundError(
        f"未找到权重文件：{weights_path}\n"
        "请先运行：python main.py --epochs 2 --no-accel "
        "--batch-size 64 --save-model"
    )

state_dict = torch.load(
    weights_path,
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(state_dict)
model.eval()
print("模型加载成功")

transform = build_transform()

test_data = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=transform,
)
parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=0)
args = parser.parse_args()

if args.index<0 or args.index>=len(test_data):
    parser.error(
        f"--index 必须在 0 到 {len(test_data) - 1} 之间"
    )

image, label = test_data[args.index]


image = image.unsqueeze(0)


with torch.no_grad():
    output = model(image)
    prediction = output.argmax(dim=1).item()

print("预测类别：",prediction)
print("真实类别：", label)