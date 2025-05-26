# FG-CROSS: Cross-Modal Retrieval in Remote Sensing via Fine-Grained Spatial Representation

This repository provides the official implementation of the paper:

> **FG-CROSS: Cross-Modal Retrieval in Remote Sensing via Fine-Grained Spatial Representation**  
> *Xu Chen, Xi Chen, et al.*  
> Submitted to *Computers & Geosciences*, 2025.

---

## 🔍 Overview

**FG-CROSS** is a cross-modal remote sensing image-text retrieval framework that integrates global CLIP features and local semantic representations via Graph Convolutional Networks (GCNs). It achieves fine-grained spatial alignment and improved retrieval performance on benchmark datasets.

---

## 📁 Project Structure

```
FG-CROSS/
├── det_json/                          # Detected object annotations
├── models/                            # Pretrained and custom model definitions
├── test/                              # Evaluation scripts and test data handling
├── clip.py                            # CLIP-based feature extraction
├── clip_gcn_image_feature_integration.py  # CLIP + GCN feature integration
├── dataloader.py                      # Dataset loading (RSITMD, RSICD, UCM)
├── evaluate.py                        # Evaluation metrics and retrieval scoring
├── infer.py                           # Inference code for retrieval
├── loss.py                            # Loss function (e.g., contrastive loss)
├── model.py                           # Model architecture
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── simple_tokenizer.py                # Tokenization utilities
├── train.py                           # Training script
├── V-Graph.ipynb                      # Jupyter notebook demo of graph structure
├── V_Graph.py                         # Graph structure extraction code
└── __init__.py                        # Package initializer
```

---

## 🛠 Environment Setup

- **Python**: 3.8
- **PyTorch**: ≥ 1.10
- Other dependencies are listed in `requirements.txt`

### Install dependencies

```bash
# (Optional) Create and activate a virtual environment
python -m venv fgcross-env
source fgcross-env/bin/activate        # For Linux/macOS
# fgcross-env\Scripts\activate         # For Windows

# Install required packages
pip install -r requirements.txt
```

---

## 📦 Datasets

The following publicly available remote sensing datasets are supported:

| Dataset | Description                                  | Link |
|---------|----------------------------------------------|------|
| RSITMD  | Remote Sensing Image-Text Multimodal Dataset | https://github.com/ucas-vg/RSITMD |
| RSICD   | Remote Sensing Image Captioning Dataset      | https://github.com/ucas-vg/RSICD-official |
| UCM     | UC Merced Land Use Dataset                   | http://weegee.vision.ucmerced.edu/datasets/landuse.html |

> 📌 Please download and extract the datasets. Then modify the paths in `dataloader.py` to point to the correct locations.

---

## 📥 Pretrained Models

Download pretrained model weights from Baidu NetDisk:

- 🔗 [https://pan.baidu.com/s/1_557f33eRK_rV5N5qHlwjA?pwd=cxcx]
- 📎 Access Code: `cxcx`

Extract the model files into the `models/` directory.

---

## 🚀 How to Use

### 1. Train the Model

```bash
python train.py
```

### 2. Run Inference

```bash
python infer.py

---> example
Similarity Score Matrix (Text-Image):
              Img1   Img2   Img3   Img4   Img5
Query 1:     0.89   0.21   0.35   0.15   0.09
Query 2:     0.22   0.94   0.13   0.33   0.11
Query 3:     0.14   0.09   0.87   0.25   0.20
Query 4:     0.18   0.22   0.31   0.91   0.28
Query 5:     0.05   0.11   0.18   0.26   0.90
```

### 3. Evaluate Retrieval Performance

```bash
python evaluate.py
```

---

This script verifies the fusion of CLIP and GCN features on sample data.

---

## 📜 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025 Xi Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📚 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{chen2025fgcross,
  title={FG-CROSS: Cross-Modal Retrieval in Remote Sensing via Fine-Grained Spatial Representation},
  author={Chen, Xi and others},
  journal={Computers \& Geosciences},
  year={2025},
  publisher={Elsevier}
}
```

---

## 📬 Contact

For questions or feedback, please contact the corresponding author:

**Mr. Xi Chen**  
📧 2024102110094@whu.edu.cn
