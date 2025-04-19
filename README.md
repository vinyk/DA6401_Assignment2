# Convolutional Neural Networks: Training from Scratch and Fine Tuning
This project aims to build a CNN and train it on the [iNaturalist dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip). The data is divided into the train and test folder, with 20% of training data being set for validation.

## Structure of the Repository
1) The code is structured into 2 separate notebooks for Part A and Part B of the assignment.
2) The notebook `Assignment_2_sweep.ipynb` includes the code for building the CNN model as specified for the assignment and running hyperparameter sweeps to find the best set.
3) The best model is then tested to find the test metrics (loss and accuracy).
4) The notebook `Assignment_2_ft.ipynb` includes the code for fine-tuning a pre-trained model and logging the performance metrics.

## Running the notebooks
1) Both the notebooks have been made to run cell by cell.
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
2) The data is divided into training and testing data, with 20% of the training data being set for validation.
3) The best hyperparameter set was found to be:
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
5) The best test accuracy with this model was **40%**.

## Part B: Fine-tuning a Pre-trained Model
1) `Resnet-50` was chosen as the pre-trained model.
2) To fine-tume the model, I selected the method of freezing the parameters of all layers and training only the final layer.
3) For 10 epochs, this gives **training and validation accuracy** of **71% and 67%** respectively.
