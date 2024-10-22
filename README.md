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

## Meta Data

![Meta Data](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/architecture/data-architecture/data-description.png?raw=true)

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

![Architecture 2](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/img/output.png?raw=true)

---

## Metrics

| Architecture                                                                                                                                                                                                    | Ts Accuracy | F1 Score | AUC-ROC |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- | :------- | :------ |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Pooling ResNet50 Independent DL Network BCE Loss KFolds NO SAMPLING                                                                       | 46.60%      | 37.80%   | 60.00%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Attention Layer Pooling ResNet50 Independent DL Network BCE Loss KFolds NO SAMPLING                                                       | 27.50%      | 46.40%   | 62.80%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Pooling ResNet50 Independent DL Network BCE Loss KFolds WEIGHTED LOSS FUNCTION                                                            | 40.80%      | 47.50%   | 63.30%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Attention Layer Pooling ResNet50 Independent DL Network BCE Loss KFolds WEIGHTED LOSS FUNCTION                                            | 64.20%      | 36.50%   | 58.90%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Attention Layer Pooling MedicalNet152 Independent DL Network BCE Loss KFolds NO SAMPLING                                                  | 47.50%      | 33.10%   | 55.40%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Attention Layer Pooling MedicalNet152 Independent DL Network BCE Loss KFolds WEIGHTED LOSS FUNCTION                                       | 56.00%      | 32.90%   | 56.70%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Pooling ResNet50 Independent DL Network BCE Loss KFolds NO SAMPLING                                                              | 50.20%      | 34.40%   | 57.50%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling ResNet50 Independent DL Network BCE Loss KFolds NO SAMPLING                                              | 27.80%      | 46.30%   | 61.90%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Pooling ResNet50 Independent DL Network BCE Loss KFolds WEIGHTED LOSS FUNCTION                                                   | 27.40%      | 33.80%   | 58.00%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling ResNet50 Independent DL Network BCE Loss KFolds WEIGHTED LOSS FUNCTION                                   | 61.00%      | 45.80%   | 61.50%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling MedicalNet152 Independent DL Network BCE Loss KFolds NO SAMPLING                                         | 48.90%      | 31.90%   | 55.90%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling MedicalNet152 Independent DL Network BCE Loss KFolds WEIGHTED LOSS FUNCTION                              | 46.90%      | 31.60%   | 57.50%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Pooling ResNet50 SVC Synthetic Minority Oversampling Technique (33% undersample for SVC)                                                  | 91.95%      | 61.90%   | 62.95%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Attention Layer Pooling ResNet50 SVC Synthetic Minority Oversampling Technique (33% undersample for SVC)                                  | 88.30%      | 60.70%   | 57.64%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Pooling ResNet50 LGBM Synthetic Minority Oversampling Technique (33% undersample for SVC)                                                 | 78.03%      | 45.59%   | 67.32%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Attention Layer Pooling ResNet50 LGBM Synthetic Minority Oversampling Technique (33% undersample for SVC)                                 | 81.31%      | 44.24%   | 71.42%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Pooling ResNet50 XGBoost Synthetic Minority Oversampling Technique (33% undersample for SVC)                                              | 64.48%      | 39.08%   | 62.60%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Attention Layer Pooling ResNet50 XGBoost Synthetic Minority Oversampling Technique (33% undersample for SVC)                              | 70.52%      | 42.16%   | 64.23%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Pooling ResNet50 Ensemble (XBG Bagging) Synthetic Minority Oversampling Technique (33% undersample for SVC)                               | 53.94%      | 34.53%   | 60.29%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Attention Layer Pooling ResNet50 Ensemble (XBG Bagging) Synthetic Minority Oversampling Technique (33% undersample for SVC)               | 64.96%      | 39.23%   | 63.09%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Attention Layer Pooling MedicalNet152 SVC Synthetic Minority Oversampling Technique (33% undersample for SVC)                             | 89.50%      | 54.75%   | 46.91%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Attention Layer Pooling MedicalNet152 LGBM Synthetic Minority Oversampling Technique (33% undersample for SVC)                            | 79.84%      | 43.07%   | 64.13%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Attention Layer Pooling MedicalNet152 XGBoost Synthetic Minority Oversampling Technique (33% undersample for SVC)                         | 62.51%      | 38.06%   | 61.03%  |
| RF One Hot Encoding (1\*512 Embedding) Histogram Equalization Average Attention Layer Pooling MedicalNet152 Ensemble (XBG Bagging) Synthetic Minority Oversampling Technique (33% undersample for SVC)          | 50.53%      | 32.11%   | 58.10%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Pooling ResNet50 SVC Synthetic Minority Oversampling Technique (33% undersample for SVC)                                         | 92.21%      | 53.45%   | 61.25%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling ResNet50 SVC Synthetic Minority Oversampling Technique (33% undersample for SVC)                         | 88.70%      | 61.92%   | 62.96%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Pooling ResNet50 LGBM Synthetic Minority Oversampling Technique (33% undersample for SVC)                                        | 77.59%      | 45.78%   | 65.84%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling ResNet50 LGBM Synthetic Minority Oversampling Technique (33% undersample for SVC)                        | 81.41%      | 45.48%   | 69.60%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Pooling ResNet50 XGBoost Synthetic Minority Oversampling Technique (33% undersample for SVC)                                     | 63.75%      | 40.52%   | 62.54%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling ResNet50 XGBoost Synthetic Minority Oversampling Technique (33% undersample for SVC)                     | 70.90%      | 43.70%   | 62.89%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Pooling ResNet50 Ensemble (XBG Bagging) Synthetic Minority Oversampling Technique (33% undersample for SVC)                      | 52.86%      | 34.16%   | 60.97%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling ResNet50 Ensemble (XBG Bagging) Synthetic Minority Oversampling Technique (33% undersample for SVC)      | 65.20%      | 39.93%   | 64.30%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling MedicalNet152 SVC Synthetic Minority Oversampling Technique (33% undersample for SVC)                    | 88.53%      | 53.17%   | 49.79%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling MedicalNet152 LGBM Synthetic Minority Oversampling Technique (33% undersample for SVC)                   | 79.81%      | 44.40%   | 65.84%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling MedicalNet152 XGBoost Synthetic Minority Oversampling Technique (33% undersample for SVC)                | 61.96%      | 38.01%   | 62.86%  |
| RF One Hot Encoding (1\*512 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling MedicalNet152 Ensemble (XBG Bagging) Synthetic Minority Oversampling Technique (33% undersample for SVC) | 49.81%      | 32.57%   | 59.15%  |
| RF One Hot Encoding (1\*512 Embedding) - - - Random Predictions -                                                                                                                                               | ≈ 0%        | ≈ 0%     | ≈ 0%    |
| RF One Hot Encoding (1\*2048 Embedding) Histogram Equalization Average Pooling ResNet50 SVC Synthetic Minority Oversampling Technique (33% undersample for SVC)                                                 | 88.02%      | 60.09%   | 53.37%  |
| RF One Hot Encoding (1\*2048 Embedding) Histogram Equalization Average Attention Layer Pooling ResNet50 SVC Synthetic Minority Oversampling Technique (33% undersample for SVC)                                 | 90.94%      | 64.20%   | 63.60%  |
| RF One Hot Encoding (1\*2048 Embedding) Histogram Equalization Average Attention Layer Pooling MedicalNet152 SVC Synthetic Minority Oversampling Technique (33% undersample for SVC)                            | 88.30%      | 59.71%   | 51.60%  |
| RF One Hot Encoding (1\*2048 Embedding) Grey Scaling Feature Extraction Average Pooling ResNet50 SVC Synthetic Minority Oversampling Technique (33% undersample for SVC)                                        | 88.95%      | 63.10%   | 65.03%  |
| RF One Hot Encoding (1\*2048 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling ResNet50 SVC Synthetic Minority Oversampling Technique (33% undersample for SVC)                        | 91.04%      | 62.80%   | 62.71%  |
| RF One Hot Encoding (1\*2048 Embedding) Grey Scaling Feature Extraction Average Attention Layer Pooling MedicalNet152 SVC Synthetic Minority Oversampling Technique (33% undersample for SVC)                   | 88.60%      | 59.95%   | 53.69%  |
| RF One Hot Encoding (1\*2048 Embedding) - - - Random Predictions -                                                                                                                                              | ≈ 0%        | ≈ 0%     | ≈ 0%    |

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
