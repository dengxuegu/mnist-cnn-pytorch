import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model import Net
from data import build_transform
from evaluate import evaluate
from torch.utils.data import random_split
from pathlib import Path
from data import build_transform,DATA_DIR
from train import train_one_epoch

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')

    parser.add_argument(
        "--test",
        action="store_true",
        help="训练结束后评估测试集",
    )

    args = parser.parse_args()

    print("本次参数是：",args)

    if args.test and not args.save_model:
        parser.error("--test 需要同时传入 --save-model，以加载本次最佳权重")

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform = build_transform()

    dataset1 = datasets.MNIST(str(DATA_DIR), train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST(str(DATA_DIR), train=False,
                       transform=transform)

    train_dataset, val_dataset = random_split(
        dataset1,
        [54000, 6000],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print("训练集：", len(train_dataset))
    print("验证集：", len(val_dataset))
    print("测试集：", len(dataset2))

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)

    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.test_batch_size,shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    best_val_loss = float("inf")
    best_model_path = Path(__file__).parent / "mnist_cnn_16ch_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,device,dry_run=args.dry_run)
        print(f"Epoch{epoch}:已处理样本的平均训练损失{train_loss:.4f}")
        print("验证机评估：")
        val_loss, val_accuracy = evaluate(model, device, val_loader)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"验证损失改善：{best_val_loss:.4f}")

            if args.save_model:
                torch.save(model.state_dict(), best_model_path)
                print("已保存最佳模型")
        scheduler.step()



    if args.test:
        if best_val_loss == float("inf"):
            raise RuntimeError("本次运行没有选出最佳模型，无法测试")

        state_dict = torch.load(
            best_model_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)

        print("最佳模型的测试集评估：")
        evaluate(model, device, test_loader)



if __name__ == '__main__':
    main()

