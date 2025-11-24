import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime
import matplotlib.pyplot as plt
import csv


current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(parent_dir)


from config import Config
from modules.ADD_MIL.add_mil import AddMIL


class TrainManager(Config):
    """
    继承自 Config 配置类，用于训练管理
    """
    def __init__(self):
        super().__init__()
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # 确保 CUDA 操作是确定性的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return
    
    def run(self):
        # 用于保存每个epoch的平均loss
        train_loss_history = []

        model = AddMIL()
        model.to(self.device)
        # 适配 2 通道输出，标签为类别索引（0或1）
        criterion = nn.CrossEntropyLoss()
        # 使用 Adam 优化器
        optimizer = optim.Adam(model.parameters(), lr=4e-4)

        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0.
            # 遍历训练集数据，计算损失并更新模型参数
            for features, labels in self.train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                A_ori, patch_contributions, logit = model(features)
                loss = criterion(logit, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 计算当前epoch的平均loss
            avg_loss = total_loss / len(self.train_loader)
            train_loss_history.append(avg_loss)
            
            # 打印每个周期的平均损失
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        # 将模型设置为评估模型，禁用Dropout和BatchNorm等训练时特有的行为
        model.eval()
        # 存储所有真实标签和正类的预测概率
        all_labels, all_probs = [], []
        # 预测正确的样本数
        correct = 0
        # 总样本数
        total = 0

        # 禁用梯度计算，减少内存占用并加速计算
        with torch.no_grad():
            # 遍历测试集
            # test_loader每次迭代返回一个批次的数据
            for features, labels in self.test_loader:
                # 将输入特征和真实标签移动到 GPU
                features = features.to(self.device)
                labels = labels.to(self.device)
                # 未经过 sigmoid 的原始预测值
                A_ori, patch_contributions, logit = model(features)
                
                # 1. 计算预测类别（取概率最大的索引）
                # 先通过softmax转换为概率，再取最大值索引（0或1）
                probs = torch.softmax(logit, dim=1)  # [batch_size, 2]，概率和为1
                predicted = torch.argmax(probs, dim=1)  # [batch_size]，预测类别（0或1）
                
                # 2. 统计正确预测数
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # 3. 提取正类（类别1）的概率，用于计算AUC
                pos_probs = probs[:, 1]  # [batch_size]，类别1的概率
                
                # 保存正类概率和真实标签
                all_probs.append(pos_probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # 拼接所有批次的结果
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # 计算AUC（使用正类概率）
        auc = roc_auc_score(all_labels, all_probs)

        # 计算准确率
        accuracy = correct / total

        # 打印结果
        print("正类预测概率：", all_probs)
        print("真实标签：", all_labels)
        print(f"Test AUC: {auc:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        # 调用保存函数
        save_dir = os.path.join(self.checkpoint_dir, self.checkpoint_prefix)
        save_model( 
            model=model, 
            epoch=self.num_epochs, 
            save_dir=save_dir,
            train_loss_history=train_loss_history,  # 传入loss历史
            test_auc=auc,  # 传入测试AUC
            test_accuracy=accuracy  # 传入测试准确率
        )
        print(f"模型和训练历史已保存")


# 保存模型（包含epoch和时间戳）
def save_model(model, epoch, save_dir="../../outputs/ADD_MIL", train_loss_history=None, test_auc=None, test_accuracy=None):
    # 自动创建保存目录（不存在则创建）
    os.makedirs(save_dir, exist_ok=True)
    logs_dir = os.path.join(save_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # 获取当前时间戳（格式：年-月-日_时-分-秒，如20251115_163045）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(logs_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # 构建文件名：前缀 + epoch + 时间戳
    filename = f"epoch{epoch}_{timestamp}.pth"
    save_path = os.path.join(save_dir, filename)
    
    # 处理多GPU模型（如果用了DDP，参数在model.module中）
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    # 保存模型参数
    torch.save(model_state, save_path)
    print(f"模型已保存至：{save_path}")

    # 保存训练历史
    if train_loss_history is not None:
        # 1. 保存为CSV文件
        csv_filename = f"loss.csv"
        csv_path = os.path.join(log_dir, csv_filename)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss'])  # 表头
            for i, loss in enumerate(train_loss_history):
                writer.writerow([i+1, loss])
        
        print(f"训练Loss已保存至：{csv_path}")
        
        # 2. 绘制Loss曲线图并保存
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, 
                 marker='o', linestyle='-', linewidth=2, markersize=5)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title(f'Training Loss Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加测试指标到图表上（如果提供）
        if test_auc is not None and test_accuracy is not None:
            plt.text(0.02, 0.98, f'Test AUC: {test_auc:.4f}\nTest Accuracy: {test_accuracy:.4f}',
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存图片
        plot_filename = f"loss.png"
        plot_path = os.path.join(log_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss曲线图已保存至：{plot_path}")
        
        # 3. 额外保存一个训练摘要文件
        summary_filename = f"summary.txt"
        summary_path = os.path.join(log_dir, summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"训练摘要\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"训练轮数: {epoch}\n")
            f.write(f"时间戳: {timestamp}\n")
            f.write(f"\n训练Loss:\n")
            f.write(f"  - 初始Loss: {train_loss_history[0]:.4f}\n")
            f.write(f"  - 最终Loss: {train_loss_history[-1]:.4f}\n")
            f.write(f"  - 最小Loss: {min(train_loss_history):.4f} (Epoch {train_loss_history.index(min(train_loss_history))+1})\n")
            
            if test_auc is not None:
                f.write(f"\n测试指标:\n")
                f.write(f"  - AUC: {test_auc:.4f}\n")
            if test_accuracy is not None:
                f.write(f"  - Accuracy: {test_accuracy:.4f}\n")
        
        print(f"训练摘要已保存至：{summary_path}")


if __name__ == "__main__":
    trainer = TrainManager()
    trainer.run()
