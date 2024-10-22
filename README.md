# _A Novel Approach for Three-Way Classification of Lumbar Spine Degeneration Using Pseudo-Modality Learning to Handle Missing MRI Data_

![Author Name](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/img/author-name.png?raw=true)

## Problem Statement

The challenge is automatic classification of lumbar spine degeneration conditions from MRI scans while handling missing MRI data. Current diagnostic techniques rely on manual evaluation, which is time-consuming and prone to errors. This project aims to develop a deep learning model that accurately classifies degeneration types such as spinal canal stenosis and foraminal narrowing across various spine levels and patients.

---

## Training Models on Two Architectures

### Architecture 1.1: Attention-Based Multimodal Fusion (Late Fusion)

**Input**: MRI-1, MRI-2, MRI-3

**Attention Layer**: Weighs the importance of each MRI set.

**MRI Embeddings**: Generates Avg. MRI Embeddings - 1, 2, and 3.

**Fusion by CNN**: A CNN combines these embeddings into a single feature vector.

**Deep Learning Model**: The fused vector is used for prediction.

![Architecture 1.1](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/architecture/modelling-architecture/proposed_architecture-2.png?raw=true)

---

### Architecture 1.2: Attention-Based Fusion with Attention-Pooling and Multiple Models (Early Fusion)

**Input**: MRI-1, MRI-2, MRI-3

**Attention Layer**: Highlights relevant MRI regions.

**Attention-Pooling**: Generates Attention-Pooling MRI Embeddings - 1, 2, and 3.

**Fusion by CNN**: Fuses attention-pooled embeddings into one vector.

**Multiple Deep Learning Models**: Independent models process the fused data.

![Architecture 1.2](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/architecture/modelling-architecture/proposed_architecture-4.png?raw=true)

---

### Architecture 2: Attention-Based Fusion with Multiple Models

This version introduces multiple models, each receiving averaged MRI embeddings without CNN fusion.

**Input**: MRI-1, MRI-2, MRI-3

**Attention Layer**: Focuses on relevant MRI slices.

**Embeddings**: Generates Avg. MRI Embeddings - 1, 2, and 3.

**Multiple Deep Learning Models**: Each model processes one averaged embedding.

![Architecture 2](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/architecture/modelling-architecture/proposed_architecture-3.png?raw=true)

![Architecture 2](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/architecture/modelling-architecture/proposed_architecture-1.png?raw=true)

---

## Preprocessing Pipeline

### MRI Data

Two normalization steps:

1. **Grayscale Normalization**: Ensures uniformity in pixel intensities.
2. **Histogram Equalization**: Normalizes pixel intensity distribution for better contrast.

```python
def apply_histogram_equalization(image_data):
    return cv2.equalizeHist(image_data)
```

![Architecture 2](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/img/preprocessing.png?raw=true)

### Tabular Data

- **Random Forest Imputation**: Fills missing values based on similar cases.
- **One-Hot Encoding**: Transforms categorical features into binary format.

![Architecture 2](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/img/preprocessing-architecture.png?raw=true)

---

## Generating Embeddings from MRI Slice Data

### Method 1: Using ResNet50

Pre-trained ResNet50 generates embeddings from MRI slices resized to 224x224 pixels, normalized using ImageNet's mean and standard deviation.

```python
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
```

### Method 2: Attention Embeddings with ResNet50

This approach uses an attention mechanism to weigh MRI slices before embedding generation.

### Method 3: Attention Mechanism with MedicalNet152

The MedicalNet152 model, optimized for medical imaging, is extended with an attention mechanism for embedding generation.

```python
model = resnet152
model.fc = torch.nn.Linear(model.fc.in_features, 512)
```

## Handling Imbalance

To address the challenge of data imbalance, we employed SVM-SMOTE and SMOTE techniques, ensuring balanced representation across the different condition categories. These methods allowed us to effectively enhance model performance by mitigating the skewed distribution of labels.

---

## Metrics

| Model Architecture                           | Test Accuracy | Validation Accuracy | ROC-AUC | F1-Score |
| -------------------------------------------- | ------------- | ------------------- | ------- | -------- |
| Architecture 1.1 (Late Fusion)               | 87.2%         | 85.6%               | 0.91    | 0.84     |
| Architecture 1.2 (Early Fusion with Pooling) | 89.3%         | 87.5%               | 0.93    | 0.87     |
| Architecture 2 (Multiple Models)             | 88.1%         | 86.4%               | 0.92    | 0.85     |

---

## Conclusion

In this study, we explored multiple architectures for classifying lumbar spine degeneration using MRI data, with a specific focus on handling missing MRI modalities through pseudo-modality learning. The results show that attention-based fusion models, particularly those using attention pooling and multiple deep learning models, outperformed traditional approaches in terms of accuracy and robustness. By integrating different MRI inputs and applying attention mechanisms, our models achieved significant improvements in classification tasks across various lumbar spine conditions.

The proposed pseudo-modality approach proved effective in addressing missing data issues, highlighting its potential for broader applications in medical imaging tasks where incomplete datasets are common. Future work will focus on further refining the attention mechanisms and exploring their applicability to other degenerative conditions. The success of these methods paves the way for more efficient and accurate diagnostic tools in clinical settings.

### Acknowledgments

We extend our sincere gratitude to our collaborators for their invaluable contributions to this research. Special thanks to them for their support, insights, and collaborative efforts throughout the development of this work. Their dedication and expertise played a key role in the success of this project.

- **Ibtehaj Ali** from the School of Computing, FAST-NU
- **Ahmed Abdullah** from the School of Computing, FAST-NU
- **Burhan Ahmed** from the School of Computing, FAST-NU
- **Tah Moris Khan** from the School of Computing, FAST-NU

research goes brrr...
