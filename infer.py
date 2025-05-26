import os
import torch
from torchvision import transforms
from PIL import Image
import clip
import clip_gcn_image_feature_integration as clip_gcn
import numpy as np

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
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
    image_encoder_ratio=(0.5, 0.5),
    use_pretrained=True
).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

# 加载图像
image_paths = [
    'test/baseballfield_665.tif',
    'test/beach_34.tif',
    'test/bareland_619.tif',
    'test/airport_504.tif',
    'test/airport_2.tif'
]
images = [preprocess(Image.open(path).convert("RGB")) for path in image_paths]
image_batch = torch.stack(images).to(device)

# 构造单位矩阵占位图结构
# 假设图结构大小为 49 x 49（7x7 patch），你需要根据实际模型结构修改
adjacency_matrices = torch.eye(20).unsqueeze(0).repeat(len(images), 1, 1).to(device)
degree_matrices = torch.eye(20).unsqueeze(0).repeat(len(images), 1, 1).to(device)

# 图像编码
with torch.no_grad():
    image_features = model.encode_image(image_batch, adjacency_matrices, degree_matrices)
    image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)

# 文本输入
sentences = [
    "There is a baseball field beside the green amusement park around the red track.",
    "A green baseball field adjacent to the playground and Red Square.",
    "There is a long path in the field next to the red playground",
    "The green playground around the red runway is a baseball field.",
    "The green baseball field is adjacent to the playground and the red playground."
]
tokenized = clip.tokenize(sentences).to(device)

# 文本编码
with torch.no_grad():
    text_features = model.encode_text(tokenized)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

# 计算相似度矩阵
similarity = text_features @ image_features.T  # 5 x 5

# 输出结果，带图像文件名
print("\nSimilarity Matrix:")
print(" " * 25 + "\t".join([os.path.basename(p) for p in image_paths]))
for i, sentence in enumerate(sentences):
    sims = "\t".join([f"{sim:.4f}" for sim in similarity[i]])
    print(f"[{i+1}] {sentence[:40]:<40} {sims}")
