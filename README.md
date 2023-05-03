# CS577_final_project_RobustQA

# Background

# fds

## Preparing data

This project was built using [Datasets](https://github.com/huggingface/datasets) for retrieving data and data preprocessing.

The class of **datasets.DatasetDict** is the dataloader to handle all the pubilc data from [Hugging Face](https://huggingface.co/))

### Load data

An example for a Q&A dataset
```
load_dataset("squad")
```
```
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 87599
    })
    validation: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 10570
    })
})
```

### Preprocess data

We divided the data into three types:
1. 'train' is the training data for model training
2. 'validation' or 'dev' is the validating data for hyperparameter tuning.
3. 'test' is the testing data for performance evaluation.

Preprocssing Proceduer
1. Tokenization(by pretrained tokenizers)
2. Extraction


This project adopted 

## Training Loop

## Evaluation

# Experiment
