import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from model import VisionTransformer, Transformer, LayerNorm
import clip
import V_Graph as vg

class ModifiedCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: int,
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 use_pretrained: bool = True,
                 image_encoder_ratio: tuple = (0.5, 0.5)):  # Adjusted parameter
        super().__init__()

        self.context_length = context_length  # 77
        self.embed_dim = embed_dim
        self.image_encoder_ratio = image_encoder_ratio

        # Load original CLIP model
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim
        )
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        self.gcn1 = GCNConv(embed_dim, embed_dim)
        self.gcn2 = GCNConv(embed_dim, embed_dim)
        self.gcn3 = GCNConv(embed_dim, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.vision_width = vision_width
        self.initialize_parameters()

        if use_pretrained:
            self.load_pretrained_weights()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def load_pretrained_weights(self):
        # Load pretrained weights for visual and text models
        pretrained_model = clip.load('ViT-B/32', device='cpu')[0]
        self.visual.load_state_dict(pretrained_model.visual.state_dict())
        self.transformer.load_state_dict(pretrained_model.transformer.state_dict())
        self.token_embedding.load_state_dict(pretrained_model.token_embedding.state_dict())
        self.positional_embedding.data = pretrained_model.positional_embedding.data
        self.ln_final.load_state_dict(pretrained_model.ln_final.state_dict())
        self.text_projection.data = pretrained_model.text_projection.data
        self.logit_scale.data = pretrained_model.logit_scale.data

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def encode_image_gcn(self, adjacency_matrix, degree_matrix):
        if len(degree_matrix.shape) == 3:
            degree_matrix = degree_matrix[:, 0, :]

        # Convert adjacency_matrix to edge_index format for GCNConv
        edge_index, edge_weight = dense_to_sparse(adjacency_matrix)

        # Use degree_matrix as the input features for the GCN
        node_features = degree_matrix.contiguous().view(-1, 1).expand(-1, self.embed_dim)

        # Process adjacency and degree matrices through GCN layers
        gcn_output = F.relu(self.gcn1(node_features, edge_index, edge_weight))
        gcn_output = F.relu(self.gcn2(gcn_output, edge_index, edge_weight))
        gcn_output = F.relu(self.gcn3(gcn_output, edge_index, edge_weight))

        # Pass through MLP
        gcn_output = self.mlp(gcn_output)

        return gcn_output

    def encode_image(self, image, adjacency_matrix, degree_matrix):
        # 提取 ViT 特征
        x = self.visual (image)

        # 将 adjacency_matrix 转换为 edge_index 格式
        edge_index, edge_weight = dense_to_sparse (adjacency_matrix)

        # 使用 degree_matrix 作为 GCN 的输入特征
        node_features = degree_matrix.contiguous ().view (-1, 1).expand (-1, self.embed_dim)

        # 经过 GCN 层处理 adjacency 和 degree 矩阵
        gcn_output = F.relu (self.gcn1 (node_features, edge_index, edge_weight))
        gcn_output = F.relu (self.gcn2 (gcn_output, edge_index, edge_weight))
        gcn_output = F.relu (self.gcn3 (gcn_output, edge_index, edge_weight))

        # 进行平均池化以匹配 batch size 维度
        gcn_output = gcn_output.mean (dim=0, keepdim=True)
        gcn_output = gcn_output.expand (image.size (0), -1)  # 将节点特征扩展为与批次相匹配

        # 通过 MLP
        gcn_output = self.mlp (gcn_output)

        # 合并 ViT 和 GCN 特征
        combined_features = self.image_encoder_ratio[0] * x + self.image_encoder_ratio[1] * gcn_output  # 默认权重
        return combined_features

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.text_projection.dtype)
        x = x + self.positional_embedding.type(self.text_projection.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.text_projection.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text, adjacency_matrix, degree_matrix):
        image_features = self.encode_image(image, adjacency_matrix, degree_matrix)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text