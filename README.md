# Convolutional Neural Networks: Training from Scratch and Fine Tuning
This project aims to build a CNN and train it on the [iNaturalist dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip). The data is divided into the train and test folder, with 20% of training data being set for validation.
A `requirements.txt` file has been provided that specifies all the libraries used for the project.

## Table of Contents

- [Structure of the Repository](#structure-of-the-repository)
- [Running the notebooks](#running-the-notebooks)
- [Part A: Training from Scratch](#part-a-training-from-scratch)
- [Part B: Fine-tuning a Pre-trained Model](#part-b-fine-tuning-a-pre-trained-model)
  - [1) Freeze All But Final Layer](#1-freeze-all-but-final-layer)
  - [2) Freeze Up to a Certain Block](#2-freeze-up-to-a-certain-block)
- [Sample Results](#sample-results)

## Structure of the Repository
1) The code is structured into 2 separate notebooks for Part A and Part B of the assignment.
2) The notebook `Assignment_2_Sweep.ipynb` includes the code for building the CNN model as specified for the assignment and running hyperparameter sweeps to find the best set.
3) The best model is then tested to find the test metrics (loss and accuracy).
4) The notebooks `ft1.ipynb` and `ft2.ipynb` discuss two different strategies for fine-tuning a pre-trained model and logging the performance metrics.

## Running the notebooks
1) All the notebooks have been designed to run cell by cell.
2) Google Colab was used for this purpose with the cloud T4 GPU access for training and inference.
3) The data needs to be first unzipped, this part of the code can be changed with accordance to the path of the file on your local machine.
```
import zipfile
import os

zip_path = "/content/drive/MyDrive/nature_12K.zip"
extract_dir = "/content/nature"

os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
```

## Part A: Training from Scratch
1) The CNN model consists of consisting of 5 convolution layers. Each convolution layer is followed by an activation and a max-pooling layer. A dense layer follows 5 such conv-activation-maxpool blocks. The output layer contains 10 neurons (1 for each of the 10 classes).
2) The data is divided into training and testing data, with 20% of the training data being set for validation using the `INaturalistDataModule` class.
3) A `LitCNN' class has been designed that includes the CNN design along with the methods for calculating the performance metrics (losses and accuracies).
4) The hyperparameter sweeps have been then run keeping the number of epochs to be 10.
5) The best hyperparameter set was found to be:
```
best_config = {
    "lr": 0.01,
    "batch_size": 64,
    "dropout": 0.3,
    "activation": "gelu",
    "filters": [32, 64, 128, 128, 256],
    "kernel_size": 5,
    "dense_neurons": 256,
    "augment": True,
    "fp16": True
}
```
4) The highest **validation accuracy** achieved was **35.45%.**.
5) The best test accuracy with this model was **31.41%**.
6) This model is available as `best_model.pth`.

## Part B: Fine-tuning a Pre-trained Model
1) `Resnet-50` was chosen as the pre-trained model.
2) Similar to the `Assignment_2_Sweep.ipynb` notebook, the dataset has been prepared using the same `INaturalistDataModule` class.
3) The `FineTuner` class includes the methods for calculating all performance metrics.
4) To fine-tume the model, I used two different strategies:

### 1) Freeze All But Final Layer
1) Freeze all layers: for `param in model.parameters(): param.requires_grad = False`
2) Replace final layer (classifier head) with a new one suited to iNaturalist (e.g., 10 classes).
3) Train only the final layer.
4) For 10 epochs, this gives **training and validation accuracy** of **71% and 67%** respectively.
5) The code for this approach is available in `ft1.ipynb`.

### 2) Freeze Up to a Certain Block
1) Freeze early layers (e.g., here, till the second block).
2) Fine-tune higher-level features (closer to task-specific decision making i.e. 3rd and 4th layers).
3) Use a smaller learning rate for pre-trained layers and higher for the new head.
4) For 10 epochs, this gives **training and validation accuracy** of **100% and 75%** respectively.
5) The code for this approach is available in `ft2.ipynb`.

## Sample Results
