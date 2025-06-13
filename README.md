# 🔍 DINOv2-Based Image Retrieval

## 📌 Overview

This project implements a deep learning-based image retrieval system that returns visually similar images from a gallery, given a query image. It is based on a fine-tuned DINOv2-base backbone and uses cosine similarity between learned embeddings for retrieval with triplet loss training.

---

## 🚀 Key Features

- **Powerful Backbone**: [DINOv2-base](https://github.com/facebookresearch/dinov2) from Facebook Research
- **Triplet Loss Training**: Margin-based triplet loss with margin=0.3 for effective embedding learning
- **Selective Fine-tuning**: Only last layers (10, 11) and layer normalization are trainable
- **Advanced Retrieval Techniques**:
  - Test-Time Augmentation (TTA) with horizontal flipping
  - L2-normalized embeddings for cosine similarity
  - Stratified train/validation split
- **Comprehensive Evaluation**: Multiple metrics including mAP@k, Precision@k, and Top-k Accuracy

---

## 🧠 Technical Details

### Model Architecture

The system uses a transformer-based architecture:

1. **Backbone**: DINOv2-base Vision Transformer provides robust self-supervised representations
2. **Feature Extraction**: [CLS] token from the final hidden state (768 dimensions)
3. **Normalization**: L2 normalization for cosine similarity-based retrieval
4. **Selective Training**: Only the last transformer layers are fine-tuned to preserve pre-trained features

### Training Strategy

The training process leverages several advanced techniques:

1. **Loss Function**: Triplet Margin Loss with margin=0.3 and L2 distance
2. **Data Sampling**: Custom TripletDataset for online triplet mining
3. **Optimization**: AdamW optimizer with learning rate 1e-4 and weight decay 5e-4
4. **Mixed Precision**: Automatic Mixed Precision (AMP) for efficient training
5. **Data Augmentation**: Random resized crop, horizontal flip, and normalization
6. **Validation**: mAP@10 metric for model selection and early stopping

### Advanced Retrieval Pipeline

For optimal retrieval results, the pipeline includes:

1. **Test-Time Augmentation (TTA)**: Averaging embeddings from original and horizontally flipped images
2. **Cosine Similarity**: Efficient similarity computation between query and gallery embeddings
3. **Top-K Retrieval**: Configurable number of most similar images returned

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/stanghee/dino-image-retrieval.git
cd dino-image-retrieval
```

2. Install dependencies (Use Python 3.12+):

```bash
pip install -r requirements.txt
```

3. Prepare the dataset:

The dataset should have the following structure:
```
your_dataset/
├── train/                        # Training images, with subfolders for each class
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/                         # Test images
    ├── query/                    # Query images (flat structure)
    └── gallery/                  # Gallery images (flat structure)
```

4. Update the dataset path in the notebook:
   
Open `Dino_image_retrieval.ipynb` and update the `DATA_DIR` variable:
```python
DATA_DIR = "your_dataset_path"
```

---

## 🚀 Usage

### Training and Inference Pipeline

1. **Run the main training notebook**:
   ```bash
   jupyter notebook Dino_image_retrieval.ipynb
   ```
   This will:
   - Load and preprocess the dataset with stratified splits
   - Train the DINOv2 model with triplet loss
   - Perform image retrieval on test set
   - Save results to `retrieval_results.json`

2. **Evaluate the results**:
   ```bash
   jupyter notebook Metrics.ipynb
   ```
   This will compute comprehensive metrics including mAP@k, Precision@k, and Top-k Accuracy.

### Configuration Options

Key hyperparameters in the training notebook:

```python
BATCH_SIZE    = 32        # Training batch size
NUM_EPOCHS    = 5         # Number of training epochs
LR            = 1e-4      # Learning rate
WEIGHT_DECAY  = 5e-4      # Weight decay for regularization
TOP_K         = 10        # Number of retrieved images
EMB_SIZE      = 768       # DINOv2-base embedding size
VAL_SPLIT     = 0.2       # Validation split ratio
```

---

## 👥 Contributors

- [@stanghee](https://github.com/stanghee) 
- [@lorenzoattolico](https://github.com/lorenzoattolico) 
- [@MolteniF](https://github.com/MolteniF)

---

## 📄 License

This project is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
