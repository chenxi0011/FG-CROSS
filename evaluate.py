import torch
import numpy as np
from sklearn.metrics import recall_score
import clip
from torch.utils.data import DataLoader
from dataloader import CLIPDataset
from tqdm import tqdm
import clip_gcn_image_feature_integration as clip_gcn

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_image_features = []
    all_text_features = []

    # 获取数据集中的所有图像和文本特征
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        anchor_images, positive_texts, adjacency_matrices, degree_matrices = batch
        anchor_images = anchor_images.to(device)
        positive_texts = clip.tokenize(positive_texts).to(device)
        adjacency_matrices = adjacency_matrices.to(device)
        degree_matrices = degree_matrices.to(device)

        # 提取图像特征，考虑到 GCN 的特征
        gcn_output = model.encode_image_gcn(adjacency_matrices, degree_matrices)
        anchor_features = model.encode_image(anchor_images, adjacency_matrices, degree_matrices).to(device)
        positive_features = model.encode_text(positive_texts).to(device)

        # 归一化特征
        anchor_features = torch.nn.functional.normalize(anchor_features, p=2, dim=-1)
        positive_features = torch.nn.functional.normalize(positive_features, p=2, dim=-1)

        # 收集所有图像和文本特征
        all_image_features.append(anchor_features)
        all_text_features.append(positive_features)

    # 合并所有特征
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)

    # 计算文本到图像的相似度
    text_to_image_similarities = torch.matmul(all_text_features, all_image_features.T)
    image_to_text_similarities = text_to_image_similarities.T

    # 计算 Recall@K
    recall_results = {}
    recall_results['Text-to-Image'] = calculate_recall(text_to_image_similarities)
    recall_results['Image-to-Text'] = calculate_recall(image_to_text_similarities)

    # 计算平均 Recall (Mean Recall, MR)
    mean_recall = (sum(recall_results['Text-to-Image']) / 3 + sum(recall_results['Image-to-Text']) / 3) / 2

    model.train()  # 恢复模型的训练模式
    return recall_results, mean_recall


def calculate_recall(similarities, k_values=[1, 5, 10]):
    recalls = []
    for k in k_values:
        correct = 0
        for i in range(similarities.shape[0]):
            top_k_indices = torch.topk(similarities[i], k=k).indices
            if i in top_k_indices:
                correct += 1
        recall = correct / similarities.shape[0]
        recalls.append(recall)
    return recalls
