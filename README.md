# _A Novel Approach for Three-Way Classification of Lumbar Spine Degeneration Using Pseudo-Modality Learning to Handle Missing MRI Data_

![Author Name](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/img/bird-eye.png?raw=true)

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

### Image Marks

## ![Architecture 1.1](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/img/image-marks.png?raw=true)

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

![Architecture 2](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/img/plot.png?raw=true)

| Architecture                                     | Ts Accuracy | F1 Score | AUC-ROC |  score |
| :----------------------------------------------- | :---------- | :------- | :------ | -----: |
| HIST + AAL + SVC + SMOTE + 2048 Emb              | 90.94%      | 64.2%    | 63.6%   | 69.308 |
| GSL + ResNet50 + SVC + SMOTE + 2048 Emb          | 88.95%      | 63.1%    | 65.03%  | 69.042 |
| GSL + AAL + SVC + SMOTE + 2048 Emb               | 91.04%      | 62.8%    | 62.71%  | 68.412 |
| HIST + ResNet50 + SVC + SMOTE                    | 91.95%      | 61.9%    | 62.95%  |  68.33 |
| GSL + AAL + SVC + SMOTE                          | 88.7%       | 61.92%   | 62.96%  | 67.692 |
| HIST + AAL + SVC + SMOTE                         | 88.3%       | 60.7%    | 57.64%  | 64.996 |
| GSL + ResNet50 + SVC + SMOTE                     | 92.21%      | 53.45%   | 61.25%  | 64.322 |
| GSL + MN152 + SVC + SMOTE + 2048 Emb             | 88.6%       | 59.95%   | 53.69%  | 63.176 |
| HIST + ResNet50 + SVC + SMOTE + 2048 Emb         | 88.02%      | 60.09%   | 53.37%  | 62.988 |
| HIST + AAL + LGBM + SMOTE                        | 81.31%      | 44.24%   | 71.42%  | 62.526 |
| GSL + AAL + LGBM + SMOTE                         | 81.41%      | 45.48%   | 69.6%   | 62.314 |
| HIST + MN152 + SVC + SMOTE + 2048 Emb            | 88.3%       | 59.71%   | 51.6%   | 62.184 |
| HIST + ResNet50 + LGBM + SMOTE                   | 78.03%      | 45.59%   | 67.32%  |  60.77 |
| GSL + ResNet50 + LGBM + SMOTE                    | 77.59%      | 45.78%   | 65.84%  | 60.166 |
| GSL + MN152 + LGBM + SMOTE                       | 79.81%      | 44.4%    | 65.84%  | 60.058 |
| GSL + MN152 + SVC + SMOTE                        | 88.53%      | 53.17%   | 49.79%  |  58.89 |
| HIST + MN152 + LGBM + SMOTE                      | 79.84%      | 43.07%   | 64.13%  | 58.848 |
| HIST + MN152 + SVC + SMOTE                       | 89.5%       | 54.75%   | 46.91%  | 58.564 |
| GSL + AAL + XGBoost + SMOTE                      | 70.9%       | 43.7%    | 62.89%  | 56.816 |
| HIST + AAL + XGBoost + SMOTE                     | 70.52%      | 42.16%   | 64.23%  |  56.66 |
| GSL + AAL + Independent ANN + KFolds + WLS       | 61.0%       | 45.8%    | 61.5%   |  55.12 |
| GSL + AAL + Ensemble (XBG Bagging) + SMOTE       | 65.2%       | 39.93%   | 64.3%   | 54.732 |
| GSL + ResNet50 + XGBoost + SMOTE                 | 63.75%      | 40.52%   | 62.54%  | 53.974 |
| HIST + AAL + Ensemble (XBG Bagging) + SMOTE      | 64.96%      | 39.23%   | 63.09%  |  53.92 |
| HIST + ResNet50 + XGBoost + SMOTE                | 64.48%      | 39.08%   | 62.6%   | 53.568 |
| GSL + MN152 + XGBoost + SMOTE                    | 61.96%      | 38.01%   | 62.86%  |  52.74 |
| HIST + ResNet50 + Independent ANN + KFolds + WLS | 40.8%       | 47.5%    | 63.3%   |  52.48 |
| HIST + MN152 + XGBoost + SMOTE                   | 62.51%      | 38.06%   | 61.03%  | 52.138 |
| HIST + AAL + Independent ANN + KFolds + WLS      | 64.2%       | 36.5%    | 58.9%   |     51 |
| HIST + AAL + Independent ANN + KFolds + NS       | 27.5%       | 46.4%    | 62.8%   |  49.18 |
| GSL + AAL + Independent ANN + KFolds + NS        | 27.8%       | 46.3%    | 61.9%   |  48.84 |
| HIST + ResNet50 + Ensemble (XBG Bagging) + SMOTE | 53.94%      | 34.53%   | 60.29%  | 48.716 |
| GSL + ResNet50 + Ensemble (XBG Bagging) + SMOTE  | 52.86%      | 34.16%   | 60.97%  | 48.624 |
| HIST + ResNet50 + Independent ANN + KFolds + NS  | 46.6%       | 37.8%    | 60.0%   |  48.44 |
| HIST + MN152 + Independent ANN + KFolds + WLS    | 56.0%       | 32.9%    | 56.7%   |  47.04 |
| GSL + ResNet50 + Independent ANN + KFolds + NS   | 50.2%       | 34.4%    | 57.5%   |   46.8 |
| GSL + MN152 + Ensemble (XBG Bagging) + SMOTE     | 49.81%      | 32.57%   | 59.15%  |  46.65 |
| HIST + MN152 + Ensemble (XBG Bagging) + SMOTE    | 50.53%      | 32.11%   | 58.1%   |  46.19 |
| GSL + MN152 + Independent ANN + KFolds + WLS     | 46.9%       | 31.6%    | 57.5%   |  45.02 |
| HIST + MN152 + Independent ANN + KFolds + NS     | 47.5%       | 33.1%    | 55.4%   |   44.9 |
| GSL + MN152 + Independent ANN + KFolds + NS      | 48.9%       | 31.9%    | 55.9%   |   44.9 |
| GSL + ResNet50 + Independent ANN + KFolds + WLS  | 27.4%       | 33.8%    | 58.0%   |   42.2 |

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
