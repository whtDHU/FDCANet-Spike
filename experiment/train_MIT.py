import os
import sys

from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import random

import numpy as np
import torch.optim as optim
# from Few_spike.fs_coding import replace_relu_with_fs
from dataProcess.MIT_process_k_flod import cross_validation, MITDataset
from load_data import patients

# replace_relu_with_fs()
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=9):
        super(ChannelAttention, self).__init__()

        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        # 深度卷积，用于学习每个通道的加权空间特征
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, groups=in_channels, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        # 使用全连接网络进行通道注意力的计算
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算通道注意力
        depthwise_out = self.depthwise_conv(x)
        # 平均池化
        avg_out = self.avg_pool(depthwise_out)
        max_out = self.max_pool(depthwise_out)
        avg_out = torch.flatten(avg_out, 1)
        max_out = torch.flatten(max_out, 1)
        # 对通道进行两层全连接处理
        avg_out = self.fc2(self.fc1(avg_out))
        max_out = self.fc2(self.fc1(max_out))

        # 相加两个池化结果
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels

        # 使用卷积核来提取空间注意力
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        # 计算空间注意力
        x_init = self.dconv5_5(x)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        spatial_att = torch.sigmoid(spatial_att)
        return spatial_att


class DualAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=9):
        super(DualAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        # 通道注意力
        channel_attention_map = self.channel_attention(x)

        # 将通道注意力映射扩展为与输入 x 形状相同
        channel_attention_map = channel_attention_map.unsqueeze(2).unsqueeze(3)
        output = channel_attention_map * x
        # 空间注意力
        spatial_attention_map = self.spatial_attention(output)
        out = spatial_attention_map * output
        return out


# 使用该注意力模块的模型
class FusionMIT(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(FusionMIT, self).__init__()

        # 使用双重注意力机制
        self.DualAttention = DualAttention(in_channels)

        # 定义卷积层
        self.conv1 = nn.Conv2d(18, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 7 * 14, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 使用双重注意力机制处理输入
        x = self.DualAttention(x)

        # 卷积层和池化操作
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # 全连接层
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    Dataset = "CHB-MIT"

    # 检查是否有可用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = patients
    seed_all(50)

    for indexPat in range(0, len(dataset)):
        patients_datas, patients_labels, kf = cross_validation(indexPat)

        # 初始化变量用于存储各折的累积指标
        total_best_val_acc = 0
        total_sensitivity = 0
        total_specificity = 0
        total_precision = 0
        total_f1_score = 0
        num_folds = 0  # 用于计数折数
        for k_fold, (train_index, val_index) in enumerate(kf.split(patients_datas)):
            train_data, val_data = patients_datas[train_index], patients_datas[val_index]
            train_labels, val_labels = [patients_labels[i] for i in train_index], [patients_labels[i] for i in
                                                                                   val_index]

            train_dataset = MITDataset(train_data, train_labels)
            val_dataset = MITDataset(val_data, val_labels)
            if dataset[indexPat] in ["06", "14"]:
                batch_size = 16
            else:
                batch_size = 64
            if dataset[indexPat] in ["08", "15", "16"]:
                batch_size = 8
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

            best_sensitivity = 0
            best_specificity = 0
            best_precision = 0
            best_f1_score = 0

            print(f"Fold {k_fold + 1}")
            print("训练集大小:", len(train_dataset))
            print("验证集大小:", len(val_dataset))

            import logging

            K = 118
            logging.basicConfig(level=logging.INFO)
            path = f"/e/wht_project/new_eeg_data/k_fold10/CHB-MIT"
            os.makedirs(path, exist_ok=True)

            # 创建模型
            model = FusionMIT(in_channels=18)
            model.to(device)

            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

            num_epochs = 100  # 训练的轮数
            best_val_acc = 0
            for epoch in range(num_epochs):
                model.train()  # 进入训练模式
                running_loss = 0.0
                correct_train = 0
                total_train = 0

                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

                train_accuracy = 100 * correct_train / total_train
                if train_accuracy >= 99.98:
                    break

                # 验证模型
                model.eval()  # 进入评估模式
                correct = 0
                total = 0
                TP = 0
                TN = 0
                FN = 0
                FP = 0

                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        outputs = model(x)
                        _, predicted = torch.max(outputs.data, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()
                        c = (predicted == y)
                        for i in range(predicted.shape[0]):
                            if (c[i] == 1).item() == 1:
                                if y[i] == 1:
                                    TP += 1
                                elif y[i] == 0:
                                    TN += 1
                            elif (c[i] == 1).item() == 0:
                                if y[i] == 1:
                                    FN += 1
                                elif y[i] == 0:
                                    FP += 1

                val_acc = 100 * correct / total
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # 计算当前折的各项指标
                    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    F1_score = (2 * sensitivity * precision) / (sensitivity + precision) if (
                                                                                                    sensitivity + precision) > 0 else 0
                    best_sensitivity = sensitivity
                    best_specificity = specificity
                    best_precision = precision
                    best_f1_score = F1_score

                    torch.save(model.state_dict(), f"{path}/Patient_{dataset[indexPat]}_Fold_{k_fold}.pth")

            # 累积该折的最佳指标
            total_best_val_acc += best_val_acc
            total_sensitivity += best_sensitivity
            total_specificity += best_specificity
            total_precision += best_precision
            total_f1_score += best_f1_score
            num_folds += 1
            print(f'最佳验证准确率: {best_val_acc:.2f}%')
            print(f'最佳灵敏度: {best_sensitivity * 100:.2f}%')
            print(f'最佳特异性: {best_specificity * 100:.2f}%')
            print(f'最佳精准率: {best_precision * 100:.2f}%')
            print(f'最佳F1分数: {best_f1_score * 100:.2f}%')

        # 所有折训练完成后，计算并记录平均指标
        log_file_path = f'{path}/Patient_{dataset[indexPat]}.log'

        logging.basicConfig(level=logging.INFO)
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        if num_folds > 0:
            avg_best_val_acc = total_best_val_acc / num_folds
            avg_sensitivity = total_sensitivity / num_folds
            avg_specificity = total_specificity / num_folds
            avg_precision = total_precision / num_folds
            avg_f1_score = total_f1_score / num_folds

            print(f'平均最佳验证准确率: {avg_best_val_acc:.2f}%')
            print(f'平均灵敏度: {avg_sensitivity * 100:.2f}%')
            print(f'平均特异性: {avg_specificity * 100:.2f}%')
            print(f'平均精准率: {avg_precision * 100:.2f}%')
            print(f'平均F1分数: {avg_f1_score * 100:.2f}%')

            logging.info(f'平均最佳验证准确率: {avg_best_val_acc:.2f}%')
            logging.info(f'平均灵敏度: {avg_sensitivity * 100:.2f}%')
            logging.info(f'平均特异性: {avg_specificity * 100:.2f}%')
            logging.info(f'平均精准率: {avg_precision * 100:.2f}%')
            logging.info(f'平均F1分数: {avg_f1_score * 100:.2f}%')

        # 关闭文件处理程序，以确保缓冲的日志被写入文件
        file_handler.close()
        logging.getLogger().removeHandler(file_handler)

        del train_loader, val_loader
