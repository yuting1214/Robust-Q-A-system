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
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
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

For 'train' and 'validation' data, we only have to care about ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
```
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions'],
    num_rows: 88729
})
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions'],
    num_rows: 10822
})
```

For 'test' data, we don't have the right answer, so we only care about ['input_ids', 'attention_mask']
```
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'example_id'],
    num_rows: 10822
})
```

#### Dataloader

For 'testing' data, DataLoader isn't needed.

```
from torch.utils.data import DataLoader

# Train
DatasetDict_squad_train.set_format("torch")
squad_train_dataloader = DataLoader(
    DatasetDict_squad_train,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=8,
)

# Validation
DatasetDict_squad_val.set_format("torch")
squad_val_dataloader = DataLoader(
    DatasetDict_squad_val,
    shuffle=False,
    collate_fn=default_data_collator,
    batch_size=8,
)

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
```

## Training Loop

```python
train_model(model, dataloaders, raw_test_dataset_dict, Token_test_dataset_dict, optimizer,
                device, path, num_epochs=5) -> torch.model:
    """
   
    :param model: model to be trained
    :type class_arg: transformers.models or torch.nn.Module
    
    :param dataloaders: data in the format of {'train':train_data_loader, 'val:val_data_loader}
    :type dict_arg: dict
    
    :param
    :type 
      :param
    :type 
      :param
    :type 
      :param
    :type 
  
    """
```

## Evaluation

# Experiment
