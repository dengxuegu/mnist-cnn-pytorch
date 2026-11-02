from torchvision import transforms
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
def build_transform():
    # 返回项目使用的预处理流程
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])