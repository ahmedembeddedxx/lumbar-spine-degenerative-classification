# **A Novel Approach for Three-Way Classification of Lumbar Spine Degeneration Using Pseudo-Modality Learning to Handle Missing MRI Data**

## Problem Statement
 The challenge addressed in this research is the automatic classification of lumbar  spine  degeneration  conditions  from  MRI  scans and handle the missing MRI diseases effectively. The currently existing diagnostic techniques depends on manually labeling and evaluation that are not only time consuming may also have human error.   The  goal of this project  is  to  build  a  deep  learning-based  classification model that can, at best, accurately identify and classify different degeneration types of spine,  such  as  spinal  canal  stenosis  and  foraminal  narrowing,  at various spine levels and multiple patients





## Preprocessing Pipeline

Prior to usage of the data as input for the model, some preprocessing steps should be performed on both MRIs and tabular.

Steps in the Pipeline:

###MRI Data:

To summarize the extraction process: The MRI data goes through two main steps of normalization they are;

Greyscale Normalization: This is for Uniformity of the pixel intensity values.


Histogram Equalization : It is used to normalize pixel intensity distribution, thus enhancing image contrast.
 ```python
    def apply_histogram_equalization(image_data):
        equalized_image = cv2.equalizeHist(image_data)
        return equalized_image
```
The normalized outputs are saved as. The data set was saved in .npy format to use easily in the machine learning models.

###Tabular Data:

The tabular data that could be demographic or clinical input processed separately. This involves:

####Implementing Random Forest Imputation:
 The process of filling in missing values using similar cases with the help of a random forest model.
 ```python
 RF = RandomForestClassifier()
            RF.fit(X_train, y_train)
            
            predicted_values = RF.predict(X_test)
            
            df.loc[df[col].isnull(), col] = predicted_values

    for col in categorical_columns:
        df[col] = label_encoders[col].inverse_transform(df[col].astype(int))

    df.insert(0, 'study_id', study_id)
```

####One Hot Encoding :
The process of turning categorical features into binary format with the help of coded numbers so that machine learning algorithm will be able to do a better job. This saves the final file as a. csv.

###Diagram:
![Alt text](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/be7af5f22013766d4c5c44780c73412e8dbeec72/img/preprocessing-architecture.png?raw=true)

#Generating Embeddings from MRI Slice Data
##Overview
This paper details three approaches for creating embeddings from MRI slice data. These embeddings act like a condensed version of the image data. They can be beneficial for classification. clustering and retrieval tasks etc. The former is based on a pre-trained ResNet50 model, and the latter uses attention-based embeddings with ResNet50. In the third method they used an attention mechanism with a MedicalNet152 model.


##Method 1: Using ResNet50
The first approach utilizes a pre-trained ResNet50 model from PyTorch for generating feature embeddings from MRI slices.
```python
model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

```
###Image Preprocessing:

MRI slice images are resized to 224x224 pixels.Images are normalize using ImageNet’s mean and standard deviation values in order to match the conditions under which the model was trained.
```python
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
```
###Loading and Processing MRI Slices:
For each patient and series, the MRI slices ,which are soterd as Numpy arrays , are loaded from directory.  If a slice is two-dimensional, it is expanded into three channels (RGB format). On the other hand, if it has only one channel, the channel is repeated thrice.
```python
 slice_file in os.listdir(series_path):
            if slice_file.endswith('.npy'):
                slice_path = os.path.join(series_path, slice_file)
                slice_data = np.load(slice_path)

                if slice_data.ndim == 2:
                    slice_data = np.stack([slice_data] * 3, axis=0)
                elif slice_data.ndim == 3 and slice_data.shape[0] == 1:
                    slice_data = np.repeat(slice_data, 3, axis=0)

```
###Generating Embeddings:

Each preprocessed slice is passed through the ResNet50 model to generate embeddings. The gradient calculation is disabled to save memory during this process.
```python
with torch.no_grad():
   embedding = model(input_tensor).view(-1)
   embeddings.append(embedding.numpy())

```
###Averaging Embeddings:

If multiple slices are processed for a series, the embeddings are averaged to create a single vector representing the entire series.
```python
if embeddings:
   average_embedding = np.mean(np.vstack(embeddings), axis=0)
   embedding_dict = {f'{i}': average_embedding[i] for i in range(512)}
   embedding_dict.update({'study_id': patient_id, 'series_id': series_id})
   results.append(embedding_dict)

```


###Saving Results:

The embeddings, along with the patient and series identifiers, are saved to a CSV file.
The state dictionary of the model (weights) is also saved for future use.
```python
results_df = pd.DataFrame(results)
    results_df.to_csv(f'{dir_name}/final_embeddings.csv', index=False)
    torch.save(model.state_dict(), f'{dir_name}/model_embeddings.pth')
```


##Method 2: Using Attention Embeddings with ResNet50
The second method incorporates an attention mechanism to enhance the generation of embeddings from MRI slices. This method builds on the ResNet50 model, adding an attention layer to weigh the importance of different slices.

###Model Setup:

The ResNet50 model is loaded similarly to Method 1, but the final fully connected layer is replaced to output 512-dimensional embeddings.
A custom attention-based embedding model is built, consisting of an attention mechanism that assigns weights to each slice before creating a final embedding.

###Loading and Processing MRI Slices:

MRI slices are loaded for each patient and preprocessed (resized to 224x224 pixels, normalized, and expanded to three channels if necessary).
```python

if slice_data.ndim == 2:
   slice_data = np.stack([slice_data] * 3, axis=0)
elif slice_data.ndim == 3 and slice_data.shape[0] == 1:
    slice_data = np.repeat(slice_data, 3, axis=0)

input_tensor = torch.from_numpy(slice_data).float().to('cuda')
input_tensor = transforms.Resize((224, 224))(input_tensor)
input_tensor = (input_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to('cuda')) / \
torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to('cuda')
input_tensor = input_tensor.unsqueeze(0)

```


###Generating Slice Embeddings:

For each MRI slice, the ResNet50 model generates an embedding. All embeddings for a patient’s series are stacked together for further processing.
```python

if embeddings:
  slice_embeddings = torch.stack(embeddings, dim=1)
  slice_embeddings = slice_embeddings.to('cuda')

```

###Applying Attention Mechanism:

The attention model calculates attention scores for each slice's embedding. These scores are then used to compute weighted embeddings, where the importance of each slice is reflected in the final output.
```python
with torch.no_grad():
  final_embedding, attention_weights = embedding_model(slice_embeddings)
  final_embedding = final_embedding.squeeze().cpu()

```

###Storing Results:

The final attention-weighted embedding for each patient’s series is stored, along with the patient identifiers.
```python
results_df = pd.DataFrame(results)
results_df.to_csv(result_path_csv, index=False)

```
##Method 3: Using Attention Mechanism in MedicalNet152
This method extends the attention mechanism to the MedicalNet152 model, which is a deep architecture specifically designed for medical imaging data.

###Model Setup:

A pre-trained MedicalNet152 model is loaded and modified to include an attention mechanism.
A linear attention layer is added to the model to compute attention scores for the embeddings generated by the base MedicalNet152 architecture.
```python
model = resnet152
    model.fc = torch.nn.Linear(model.fc.in_features, 512)
    embedding_model = MRIEmbeddingModel(model, embedding_dim=512)

    model = model.to('cuda')
    embedding_model = embedding_model.to('cuda')
```

###Image Preprocessing:

MRI slices are resized and normalized similarly to previous methods.
Single-channel slices are expanded to three channels if necessary.
```python
if slice_data.ndim == 2:
  slice_data = np.stack([slice_data] * 3, axis=0)
elif slice_data.ndim == 3 and slice_data.shape[0] == 1:
  slice_data = np.repeat(slice_data, 3, axis=0)

input_tensor = torch.from_numpy(slice_data).float().to('cuda')
input_tensor = transforms.Resize((224, 224))(input_tensor)
input_tensor = (input_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to('cuda')) / \
torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to('cuda')
input_tensor = input_tensor.unsqueeze(0)
```
###Generating Embeddings:

MRI slices are passed through the MedicalNet152 model to generate initial embeddings.
The attention layer computes scores for these embeddings, and the scores are used to produce a final weighted embedding.
```python

 torch.no_grad():
 embedding = model(input_tensor)
 embeddings.append(embedding)
```


# Training Models on Two Different Architectures
##Architecture 1:  
###Method1: Attention-Based Multimodal Fusion via CNN


####Input:
The input consists of three MRI sets  that are MRI-1, MRI-2  and MRI-3.

####Attention layer:
An Attention layer is applied to weight the importance of each input MRI set.

####MRI Embedding generation:
The attention-processed inputs generate Avg. MRI Embedding - 1, Avg. MRI Embedding - 2, and Avg. MRI Embedding - 3.

####Multimodal Fusion by CNN:
A CNN fuses these embeddings into one single feature vector.

####Deep Learning Model:
The fused vector is passed into a single deep learning model for prediction.
![Alt text](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/architecture/modelling-architecture/proposed_architecture-2.png?raw=true)

###Method 2: Attention-Based Multimodal Fusion with Attention-Pooling and Multiple Deep Learning Models
The architecture extends the attention mechanism by incorporating attention pooling and then feeding the results into multiple deep learning models for more robust prediction.

####Input:
the inputs are MRI-1, MRI-2, and MRI-3.

####Attention layer:
An Attention layer is applied to the inputs in order to  highlight the relevant MRI regions.

####Attention-Pooling embedding generation:
The attention-modified inputs are transformed into Attention Pooling MRI Embedding - 1, Attention Pooling MRI Embedding - 2, and Attention Pooling MRI Embedding - 3.

####Multimodal fusion by CNN:
A CNN fuses the attention-pooled embeddings into one unified vector representation.

####Multiple deep learning models:
the fused embeddings are then fed into multiple deep learning models, each processing the data independently.
![Alt text](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/architecture/modelling-architecture/proposed_architecture-4.png?raw=true)
##Architecture 2:
###Method : Attention-Based Fusion with Multiple Deep Learning Models
This architecture is a modification of the previous architecture by introducing multiple deep learning models in parallel, with each receiving the averaged MRI embeddings separately. The approach of Multimodal Fusion bu CNNs is not used. The architecture is described as follows:


####Input:
 Three MRI datasets: MRI-1, MRI-2 and MRI-3.

####Attention layer:
An Attention Layer is used to apply attention weights, focusing on the most relevant MRI slices.

####MRI embedding generation:
 The outputs of the attention mechanism result in Avg. MRI Embedding - 1, Avg. MRI Embedding - 2, and Avg. MRI Embedding - 3.

####Multiple deep learning models:
 Each averaged embedding is then fed into three separate deep learning models, which independently process the embeddings.

####Final Output:
 The outputs from the three models are combined to generate the final prediction.
 ![Alt text](https://github.com/ahmedembeddedxx/lumbar-spine-degenerative-classification/blob/main/architecture/modelling-architecture/proposed_architecture-3.png?raw=true)







##Metrics
The performance of embedding models were evaluated using various metrics to estimate accuracies, F1-scores and the model's capability to differentiate between classes(AUC-ROC).These metrics help us to quantify the effectiveness of various embedding techniques,image processing and classification methods applied to MRI slices.

###1. Validation accuracy:
This quantifies the percentage of samples in the validation dataset that are correctly classified.It is employed to assess how well the model fits training data that has not yet been observed.


####Key insights:
When it comes to validation accuracy,models with weighted loss functions(WLF) outperform those without sampling strategies.

Using a weighted loss function, MedicalNet152 and Histogram Equalization model got the maximum validation accuracy of 53.9%.



###2. Test accuracy:
Test accuracy is a crucial parameter for evaluating generalization performance since it quantifies the model's capacity to correctly classify novel and unseen input.

####Key insights:
The classifier with the highest test accuracy (92.21%) was the ResNet50 + Average Pooling + SVC classifier employing Gray Scaling Feature Extraction.

Models using Attention Layer Pooling combined with Support Vector Classifiers (SVC) consistently delivered strong test accuracy scores, indicating that attention mechanisms help the model focus on important MRI slices.

###3.F1-Score:
It is the harmonic mean of recall and accuracy . The macro version is very helpful for unbalanced datasets as  it calculates the F1-Score for each class separately and following with averaging the results.

####Key insights:
By using the Gray Scaling Feature Extraction, the ResNet50 + Average Pooling + SVC classifier produced the best macro F1-Score of 53.45%. This suggests that all classes have a reasonable balance between recall and accuracy.

In short,models integrated with weighted loss functions have performed better than those without them, obtaining higher overall F1-scores in various configurations.

###4. Receiver Operating Characteristic(AUC-ROC):
The model's capability  to distinguish between positive and negative classes is measured by the AUC-ROC. Positive occurrences are ranked higher in the model than negative ones, based on a higher AUC.

####Key insights:
The highest AUC-ROC score,which was 65.8% was achieved by ResNet50+Average Attention Layer Pooling +LGBM along with the Gray Scaling Feature Extraction, which suggests that this particular configuration is relatively effective at distinguishing between the classes.

In addition, attention-based pooling models generally produced higher AUC-ROC scores compared to simple pooling models that indicate that the attention mechanism improves the ability of the model to distinguish.

###5. Effect of Sampling Techniques
Different sampling strategies were used to address class imbalance, including No Sampling, Weighted Loss Functions, and SMOTE (Synthetic Minority Oversampling Technique).

####Key insights:
Models using a Weighted Loss Function generally performed better in terms of AUC-ROC and Test Accuracy compared to those without sampling. For instance, ResNet50+Weighted Loss achieved a test accuracy of 64.2% and AUC-ROC of 63.8% using Histogram equalization.
In addition, SMOTE with under sampling did not out-perform weighted loss approaches that suggests that careful loss weighting is more effective for handling class imbalance in this case.

###6. Attention Mechanism vs. Average Pooling
The use of Average Attention Layer Pooling is compared to traditional Average Pooling across multiple models.

####Key insights:
Models that incorporate attention layer pooling out-performed those using average pooling, particularly when paired with classifiers like SVC and LGBM. This signifies that the attention mechanisms allow model to focus on more relevant slices, improving both classification and  performance to distinguish .

Moreover,the combination of Attention Layer Pooling+ResNet50 showed strong performance, with high AUC-ROC values, especially when using Gray Scaling feature extraction.

###7. Ensemble models:
Some models have used ensemble techniques to enhance the performance of classification which include techniques such as XGBoost and Bagging .

####Key insights:
While ensemble models may have generally performed good, they did not out-perform SVC classifiers in the long run. For instance , ResNet50 + Attention Layer Pooling + SVC has achieved a higher AUC-ROC (65.8%) than the XGBoost + Bagging Ensemble which achieved AUC-ROC of 62.8.

## Our Approach for Handling Missing Modalities
When building models that rely on multiple types of input data (e.g., MRI-1, MRI-2, and MRI-3), there are situations where one or more modalities might be unavailable for certain patients. This missing data can hinder the model’s ability to make accurate predictions if not handled properly. To address this issue, we have implemented strategies within our multi-modal fusion by CNN architecture to maintain predictive performance even in the face of missing modalities.

###1. Modality-Agnostic Embedding Generation
In our architectures, we use attention-based embedding generation for each MRI modality before performing multi-modal fusion. By generating embeddings independently for each modality, the model ensures that the absence of one modality does not directly affect the representation of the others.

####Attention Layer:
For each available modality (e.g., MRI-1, MRI-2, MRI-3), the attention layer generates embeddings based on the most relevant slices.
Independent Embedding Generation: Since embeddings are created independently for each modality, the model can still produce meaningful embeddings from the available modalities even when one is missing.
###2. Handling Missing Modalities in Multi-Modal Fusion
When some MRI datasets are unavailable, our multi-modal fusion by CNN handles the situation in a way that minimizes the impact of the missing data:

####Flexible Fusion Mechanism:
 The CNN-based fusion mechanism is designed to accommodate different numbers of input modalities. When a modality is missing, the CNN adjusts by fusing only the available embeddings without penalizing the model's performance.

####Zero-Vector Padding for Missing Modalities:
 For instances where a modality is entirely absent, the model uses a zero-vector padding approach. A zero vector (or a vector of learned "neutral" values) replaces the missing modality's embedding, ensuring that the fusion process proceeds smoothly. This technique prevents the model from receiving biased or incorrect information due to missing data.

####Dynamic Weighting:
During the fusion process, we employ dynamic weighting to balance the contributions of each available modality. If a modality is missing, its corresponding weight is automatically set to zero, and the other modalities are weighted proportionally higher. This ensures that the absence of one modality does not disproportionately impact the final fused representation.

###3. Robustness in the DL Models
After the fusion step, the resulting multi-modal embedding is passed into one or multiple DL models, depending on the architecture. These models are trained to handle scenarios where the fused embedding may come from an incomplete set of modalities.

####Training with Missing Data:
During training, we simulate missing modalities by randomly removing certain MRI datasets from the input, allowing the deep learning models to learn how to handle incomplete data during inference. This training technique helps the models to become more robust in real-world situations where data may be incomplete.

####Regularization Techniques:
We also employ regularization techniques, such as dropout and L2 regularization, to prevent over-reliance on any single modality. This ensures that the deep learning models are not overly dependent on the presence of specific MRI datasets and can still make accurate predictions when some are missing.

###Benefits of Our Approach
Resilience to Missing Data: Our architecture is designed to be resilient when one or more modalities are missing, ensuring that the model can still generate meaningful predictions without a complete set of MRI datasets.

####Modality Flexibility:
The approach allows the model to dynamically adapt to the available data, whether it is a full set of modalities or only partial information.

####Efficient Use of Available Data:
By using a multi-modal fusion strategy with flexible handling of missing data, our model ensures that the available data is fully utilized to the best extent possible.

####Seamless Integration:
The architecture integrates seamlessly into our multi-modal fusion process, ensuring that the deep learning models can handle variable data availability without a significant drop in performance.








## Our Metrics

## Conclusion
In conclusion, our work focuses on developing robust and adaptive deep learning architectures for medical imaging, particularly in handling multiple MRI modalities. We explored different approaches, utilizing attention mechanisms and multi-modal fusion by CNN to generate meaningful embeddings from MRI data. Special emphasis was placed on addressing the common challenge of missing modalities, where our dynamic fusion and embedding strategies ensure the model's resilience. By leveraging these methods, we enable the models to make accurate predictions even with incomplete data, enhancing their real-world applicability in medical diagnostics and analysis.
