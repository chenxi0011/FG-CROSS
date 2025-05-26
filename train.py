import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import clip
from dataloader import CLIPDataset
from loss import triplet_loss
from evaluate import evaluate
from termcolor import colored
import clip_gcn_image_feature_integration as clip_gcn

# 设置训练参数
DATASET_PATH = '../data/RSITMD/images'
TRAIN_FILENAME = '../data/RSITMD/train_filename.txt'
TRAIN_CAPS = '../data/RSITMD/train_caps.txt'
JSON_FOLDER_PATH = './det_json'
BATCH_SIZE = 64  # 使用较小的批次大小
EPOCHS = 100
LR = 1e-6  # 降低学习率
MARGIN = 0.2
MAX_VIOLATION = False
GRAD_CLIP = 1.0  # 设置梯度裁剪阈值
VAL_SPLIT = 0.2  # 验证集比例
IMAGE_ENCODER_RATIO = (0.5, 0.5)  # 图像模态编码的比例
USE_PRETRAINED = True  # 是否使用预训练模型

BEST_MR = float('-inf')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 加载模型和数据集
model = clip_gcn.ModifiedCLIP(
    embed_dim=512,
    image_resolution=224,
    vision_layers=12,
    vision_width=768,
    vision_patch_size=32,
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12,
    image_encoder_ratio=IMAGE_ENCODER_RATIO,
    use_pretrained=USE_PRETRAINED
).to(device)

# 设置模型的嵌入维度
embed_dim = 512

# 转换为 float32 以减少数值误差
model = model.float()

# 冻结视觉部分权重
# for name, param in model.named_parameters():
#     if "visual" in name:
#         param.requires_grad = False

# 使用自定义的图像预处理，确保图像范围在适当区间
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

# 使用自定义的CLIP数据集类
dataset = CLIPDataset(DATASET_PATH, TRAIN_FILENAME, TRAIN_CAPS, transform, JSON_FOLDER_PATH)

# 划分训练集和验证集
train_size = int((1 - VAL_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 定义优化器
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# 开始训练
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch_idx, (anchor_images, positive_texts, adjacency_matrices, degree_matrices) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
        # Move tensors to the device
        anchor_images = anchor_images.to(device)

        # 确保 positive_texts 是一个字符串列表
        positive_texts = [text for text in positive_texts]

        # print(positive_texts)
        positive_texts = clip.tokenize(positive_texts).to(device)
        # print(positive_texts)
        adjacency_matrices = adjacency_matrices.to(device)
        degree_matrices = degree_matrices.to(device)

        # 前向传播
        gcn_output = model.encode_image_gcn(adjacency_matrices, degree_matrices)

        anchor_features = model.encode_image(anchor_images, adjacency_matrices, degree_matrices)
        positive_features = model.encode_text(positive_texts)
        # print(positive_features)
        # 数值检查
        if not torch.isfinite(anchor_features).all():
            print(f"Warning: NaN or Inf detected in anchor features at Batch {batch_idx}")
            continue
        if not torch.isfinite(positive_features).all():
            print(f"Warning: NaN or Inf detected in positive features at Batch {batch_idx}")
            continue

        # L2 归一化特征
        anchor_features = torch.nn.functional.normalize(anchor_features, p=2, dim=-1)
        positive_features = torch.nn.functional.normalize(positive_features, p=2, dim=-1)

        # 计算损失
        scores = torch.matmul(anchor_features, positive_features.T)
        print ("Scores:", scores[:5, :5])  # 只打印一小部分
        loss = triplet_loss(scores, MARGIN, max_violation=MAX_VIOLATION)

        # 数值检查
        if not torch.isfinite(loss):
            print(f"Warning: NaN or Inf detected in loss at Batch {batch_idx}")
            continue

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        # torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        total_loss += loss.item()

        # 打印损失和调试信息
        if batch_idx % 10 == 0:
            # print(f"Batch {batch_idx} - Loss: {loss.item()}")
            print(colored(f"Batch {batch_idx} - Loss: {loss.item()}", "red"))
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

    # 评估模型
    print(colored("\nEvaluating model...", "yellow"))
    recall_metrics, avg_mr = evaluate(model, val_loader, device)
    print(colored(f"Evaluation Metrics: {recall_metrics}", "cyan"))
    print(colored(f"Mean Recall (MR): {avg_mr}", "cyan"))

    # 保存最优模型
    if avg_mr > BEST_MR:
        BEST_MR = avg_mr
        torch.save(model.state_dict(), "best_model.pth")
        print(colored(f"Best model saved with MR: {BEST_MR}", "green"))

print("Training completed.")
