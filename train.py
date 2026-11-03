import torch.nn.functional as F

def train_one_epoch(model, loader, optimizer, device, dry_run=False):
    # 1. 切换为训练模式
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        # 2. 将数据移动到指定设备
        images = images.to(device)
        labels = labels.to(device)

        # 3. 清除上一批次的梯度
        optimizer.zero_grad()

        # 4. 前向传播
        output = model(images)

        # 5. 计算损失
        loss = F.nll_loss(output, labels)

        # 6. 反向传播
        loss.backward()



        # 7. 更新模型参数
        optimizer.step()

        # 8. 累加当前 batch 的总损失和样本数
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # 9. Dry Run：完成第一个 Batch 后退出
        if dry_run:
            break

    # 10. 计算整轮平均损失
    return  total_loss / total_samples

