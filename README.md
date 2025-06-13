# Vision-Language Systems: Image Captioning and Personalized Generation

A comprehensive research project exploring the bidirectional relationship between vision and language through image captioning and personalized image generation using state-of-the-art deep learning models.

## 🔬 Research Overview

This project addresses two fundamental challenges in computer vision and natural language processing:

1. **Image Captioning**: Generating natural language descriptions for visual content
2. **Personalized Image Generation**: Adapting large-scale diffusion models for customized content creation

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Contributing](#contributing) 

## ✨ Features

### Image Captioning
- Comprehensive survey and implementation of deep learning approaches
- Fine-tuned BLIP model on COCO dataset
- Support for ResNet-50 and DenseNet-201 architectures
- Integration with vision-language pretraining models
- Advanced evaluation metrics implementation

### Personalized Image Generation
- Efficient fine-tuning strategy for Stable Diffusion XL
- DreamBooth + LoRA (Low-Rank Adaptation) combination
- Stylistic fidelity preservation with semantic generalization
- Scalable solution for low-data settings

## 🏗️ Architecture

### Image Captioning Pipeline
```
Input Image → CNN Feature Extractor → RNN Decoder → Caption Generation
```

**Key Components:**
- **Feature Extraction**: ResNet-50, DenseNet-201
- **Sequence Generation**: RNN-based decoders
- **Vision-Language Models**: BLIP fine-tuning

### Personalized Generation Pipeline
```
Base Stable Diffusion XL → DreamBooth + LoRA → Fine-tuned Model → Personalized Images
```

**Key Components:**
- **Base Model**: Stable Diffusion XL
- **Adaptation**: DreamBooth methodology
- **Efficiency**: LoRA for parameter-efficient fine-tuning

## 📊 Datasets

### Image Captioning
- **MS COCO**: Primary dataset for training and evaluation
- **Flickr8k**: Additional validation dataset
- Custom preprocessing pipelines included

### Personalized Generation
- **3D Icon Dataset**: Curated collection for style adaptation
- Low-data setting optimizations

## Installation

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/vision-language-research.git
cd vision-language-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py
```

### Dependencies
```bash
pip install torch torchvision transformers
pip install diffusers accelerate
pip install datasets pillow matplotlib
pip install nltk scikit-image
pip install wandb  # For experiment tracking
```

## 💻 Usage

### Image Captioning

#### Training
```bash
# Fine-tune BLIP on COCO dataset
python train_captioning.py \
    --model blip \
    --dataset coco \
    --batch_size 32 \
    --epochs 10 \
    --lr 1e-4
```

#### Inference
```python
from models.captioning import BLIPCaptioner

# Load fine-tuned model
captioner = BLIPCaptioner.from_pretrained('./checkpoints/blip-coco-finetuned')

# Generate caption
image_path = "path/to/your/image.jpg"
caption = captioner.generate_caption(image_path)
print(f"Generated caption: {caption}")
```

### Supported Architectures

#### Image Captioning
- **BLIP** (Bootstrapped Language-Image Pre-training)
- **ResNet-50** + RNN variants
- **DenseNet-201** + attention mechanisms
- Custom vision-language fusion models

#### Image Generation
- **Stable Diffusion XL**
- **DreamBooth** adaptation
- **LoRA** (Low-Rank Adaptation)

```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.


## Contact
 **Author**:  Badri insaf , Marwa Sghir

---
