import os
import numpy as np  # 添加此行
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import V_Graph as vg  # 这里导入 V_Graph 模块

# 其余代码保持不变


def _convert_image_to_rgb(image):
    return image.convert("RGB")

# CLIP 原始图像预处理函数
def _transform(n_px=224):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class CLIPDataset (Dataset):
    def __init__(self, dataset_path, train_filename, train_caps, transform=None, json_folder_path=None):
        self.dataset_path = dataset_path
        self.transform = transform or _transform ()  # 默认使用 CLIP 的预处理
        self.json_folder_path = json_folder_path  # 新增 json_folder_path 属性
        self.data = []

        # Load image names and captions
        with open (train_filename, 'r') as f_names, open (train_caps, 'r') as f_caps:
            image_names = f_names.readlines ()
            captions = f_caps.readlines ()

        # Each image corresponds to 5 captions
        assert len (captions) == 5 * len (image_names), "Number of captions should be 5 times the number of images"

        # Expand image names 5 times and align with captions
        for i, image_name in enumerate (image_names):
            image_name = image_name.strip ()
            image_path = os.path.join (dataset_path, image_name)

            # Remove the file extension to create a JSON filename
            json_filename = os.path.splitext (image_name)[0] + ".json"
            json_file_path = os.path.join (json_folder_path, json_filename) if json_folder_path else None

            for j in range (5):  # Expand image names for each caption (5 captions per image)
                caption = captions[i * 5 + j].strip ()
                if os.path.exists (image_path) and (json_file_path is None or os.path.exists (
                        json_file_path)):  # Check if both image and JSON file exist
                    self.data.append ((image_path, caption, json_file_path))
                else:
                    print (
                        f"Warning: Either image {image_path} or its JSON file {json_file_path} does not exist and will be skipped.")

    def __len__(self):
        return len (self.data)

    def __getitem__(self, idx):
        max_attempts = 5
        attempts = 0

        while attempts < max_attempts:
            image_path, caption, json_file_path = self.data[idx]
            try:
                # Load the image using PIL and handle .tif format
                with Image.open (image_path) as img:
                    anchor_image = img.convert ("RGB")

                # Apply transformations (should be PIL at this point)
                if self.transform:
                    anchor_image = self.transform (anchor_image)

                # Load JSON data for the corresponding image if available
                adjacency_matrix = np.zeros ((20, 20), dtype=float)  # Default empty adjacency matrix
                degree_matrix = np.zeros ((20, 20), dtype=float)  # Default empty degree matrix

                if json_file_path:
                    json_data = vg.load_json (json_file_path)
                    if json_data is None:
                        raise ValueError (f"JSON data for {image_path} is missing or failed to load.")
                    # Use the existing function to create adjacency and degree matrices
                    adjacency_matrix, degree_matrix = vg.build_matrices (json_file_path)

                # Convert adjacency and degree matrices to tensors
                adjacency_matrix_tensor = torch.tensor (adjacency_matrix, dtype=torch.float32)
                degree_matrix_tensor = torch.tensor (degree_matrix, dtype=torch.float32)

                # Check if image tensor has any NaN or Inf values
                if not torch.isfinite (anchor_image).all ():
                    raise ValueError (f"Image {image_path} contains NaN or Inf values and will be skipped.")

                return anchor_image, caption, adjacency_matrix_tensor, degree_matrix_tensor

            except Exception as e:
                print (f"Error loading data (attempt {attempts + 1}/{max_attempts}) for image {image_path}: {e}")
                idx = (idx + 1) % len (self.data)
                attempts += 1

        # If all attempts fail, raise an error to avoid infinite recursion
        raise RuntimeError (f"Unable to load a valid image and JSON data after {max_attempts} attempts.")
