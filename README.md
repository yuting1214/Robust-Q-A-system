# CS577_final_project_RobustQA

# Background

# fds

## Preparing data

This project was built using [Datasets](https://github.com/huggingface/datasets) for retrieving data and data preprocessing.

The class of **datasets.DatasetDict** is the dataloader to handle all the pubilc data from [Hugging Face](https://huggingface.co/)

### Load data

An example for a Q&A dataset
```
raw_squad=load_dataset("squad")
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

Preprocssing Procedures
1. Tokenization(by pretrained tokenizers)
2. Extraction
3. Load in [Pytorch Dataloader](https://pytorch.org/docs/stable/data.html)

The default tokenizer is
```
from transformers import AutoTokenizer

 
```

#### Tokenization

For 'train' and 'validation' data, transform the data by the following function
```
from utils import preprocess_train_dev_data, preprocess_test_data, example_prepare

DatasetDict_squad_train = example_prepare(raw_squad, 'train', preprocess_train_dev_data, tokenizer)
DatasetDict_squad_val = example_prepare(raw_squad, 'validation', preprocess_train_dev_data, tokenizer)
```

For 'test' data, transform the data by the following function
```
# In default "squad" dataset, it only has 'validation' and we use it for testing.
DatasetDict_squad_test = example_prepare(raw_squad, 'validation', preprocess_test_data, tokenizer)
```

#### Extraction

For 'train' and 'validation' data, we have 
```
```

```
```


## Usage

```python
def my_function(string_arg: str, int_arg: int) -> None:
    """
    [Insert a brief description of the function here]
    
    :param string_arg: [Insert a brief description of string_arg here]
    :type string_arg: str
    
    :param int_arg: [Insert a brief description of int_arg here]
    :type int_arg: int
    """
    [Insert the function code here]



This project adopted 

## Training Loop

## Evaluation

# Experiment
