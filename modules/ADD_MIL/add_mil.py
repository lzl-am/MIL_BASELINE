import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class AddMIL(nn.Module):
    """
    Additive Multiple Instance Learning (AddMIL)
    
    核心公式: g(x) = Σ ψ_p(α_i · f(x_i))
    
    关键特性:
    1. 每个patch产生class-wise的贡献分数
    2. 最终预测是各patch贡献的加权和
    3. 可区分excitatory(正)和inhibitory(负)贡献
    """
    def __init__(self, feature_dim=1024, inner_dim=512, num_classes=2, dropout=0):
        super().__init__()
        # 多头注意力机制
        # TODO：当前只支持n_heads=1
        self.n_heads = 1
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, inner_dim),
            nn.Tanh(),
            nn.Linear(inner_dim, self.n_heads)
        )

        # 对每个patch独立产生class-wise贡献
        self.patch_classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )

        self.apply(initialize_weights)

    def forward(self, features):
        # 计算注意力权重
        attn_weights = self.attention(features)  # (batch_size, n_patches, n_heads)
        # 保存原始注意力权重
        A_ori = attn_weights.clone()
        # 对注意力权重应用softmax归一化
        A_normalized = F.softmax(attn_weights, dim=1)  # (batch_size, n_patches, n_heads=1)

        # attention加权特征(广播乘法) → α_i · f(x_i)
        weighted_features = A_normalized * features  # (batch_size, n_patches, feature_dim)
        
        # 分类加权特征 → ψ_p(α_i · f(x_i))
        patch_contributions = self.patch_classifier(weighted_features)  # (batch_size, n_patches, num_classes)

        # 求和 → Σ ψ_p(α_i · f(x_i))
        logits = torch.sum(patch_contributions, dim=1)  # (batch_size, num_classes)

        return A_ori, patch_contributions, logits


if __name__ == "__main__":
    input = torch.randn(1, 10077, 1024).cuda()
    model = AddMIL().cuda()
    A_ori, patch_contributions, logits = model(input)
    print(A_ori.shape)
    print(patch_contributions.shape)
    print(logits.shape)
    print("开始前向传播预热...")
    for _ in tqdm(range(2), desc="前向预热", unit="次"):
        model(input)
    torch.cuda.synchronize()  # 确保GPU操作完成
    
    print("开始前向传播正式测试...")
    start_time = time.time()
    for _ in tqdm(range(2), desc="前向测试", unit="次"):
        model(input)
    torch.cuda.synchronize()
    forward_total = time.time() - start_time
    forward_avg_ms = (forward_total / 2) * 1000

    print("开始反向传播预热...")
    for _ in tqdm(range(2), desc="反向预热", unit="次"):
        A_ori, patch_contributions, logits = model(input)
        logits.sum().backward()  # 模拟损失函数反向传播
        model.zero_grad()  # 清除梯度，避免累积影响
    torch.cuda.synchronize()

    print("开始反向传播正式测试...")
    start_time = time.time()
    for _ in tqdm(range(2), desc="反向测试", unit="次"):
        A_ori, patch_contributions, logits = model(input)
        logits.sum().backward()  # 模拟损失函数反向传播
        model.zero_grad()
    torch.cuda.synchronize()
    backward_total = time.time() - start_time
    backward_avg_ms = (backward_total / 2) * 1000

    print(f"前向传播平均时间: {forward_avg_ms:.4f} ms")
    print(f"反向传播平均时间: {backward_avg_ms:.4f} ms")