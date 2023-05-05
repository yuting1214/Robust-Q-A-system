import os
import sys
import time
import copy
import numpy as np
import pandas as pd
import collections
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, BertForQuestionAnswering
import evaluate

# Model training
## (1) General Train loop
def train_model(model, dataloaders, raw_test_dataset_dict, Token_test_dataset_dict, optimizer,
                device, path, num_epochs=3): 
    since = time.time()
    #best_model_wts = copy.deepcopy(model.state_dict())
    epoch_loss_train = []
    epoch_loss_val = []
    metric_val = [] # [F1, exact match]
    best_loss_val = sys.maxsize
    num_update_steps_per_epoch = len(dataloaders['train'])
    num_training_steps = num_epochs * num_update_steps_per_epoch
    progress_bar = tqdm(range(num_training_steps))
    model_file_path = os.path.join(path, "model.pt")
    metric_file_path = os.path.join(path, "metric.csv")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                start_logits = []
                end_logits = []
                
            running_loss = 0.0
            dataloader = dataloaders[phase] # Class: torch.utils.data.dataloader.DataLoade
            dataset_sizes = len(dataloader.dataset)
            # Iterate over data.
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(**batch)
                    loss = outputs.loss
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                    else:
                        start_logits.append(outputs.start_logits.cpu().numpy())
                        end_logits.append(outputs.end_logits.cpu().numpy())
                running_loss += loss.item()
                
            epoch_loss = running_loss / dataset_sizes                     
            if phase == 'train':
                epoch_loss_train.append(epoch_loss)
                print(f'{phase} Loss: {epoch_loss:.4f}')
            else:
                epoch_loss_val.append(epoch_loss)
                start_logits = np.concatenate(start_logits)
                end_logits = np.concatenate(end_logits)
                metric_value = compute_metrics(start_logits, end_logits, raw_test_dataset_dict, Token_test_dataset_dict)
                metric_val.append([metric_value['f1'], metric_value['exact_match']])
                print(f'{phase} Loss: {epoch_loss:.4f} Metric: {metric_value}')
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss_val:
                best_loss_val = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss}, model_file_path)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # Export training progress
    loss_df = pd.DataFrame({'epoch': range(1, num_epochs+1), 'epoch_loss_train':epoch_loss_train, 'epoch_loss_val':epoch_loss_val})
    metric_df =  pd.DataFrame(metric_val)
    metric_df.columns = ['F1', 'EM']
    result_df = pd.concat([loss_df, metric_df], axis=1)
    result_df.to_csv(metric_file_path, index=False)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

## (2) MoE Train loop
def train_model_moe(model, dataloaders, raw_dataset_dict, Token_dataset_dict, optimizer, device, path, num_epochs=3): 
    since = time.time()
    #best_model_wts = copy.deepcopy(model.state_dict())
    epoch_loss_train = []
    epoch_loss_val = []
    metric_val = [] # [F1, exact match]
    best_loss_val = sys.maxsize
    num_update_steps_per_epoch = len(dataloaders['train'])
    num_training_steps = num_epochs * num_update_steps_per_epoch
    progress_bar = tqdm(range(num_training_steps))
    model_file_path = os.path.join(path, "model.pt")
    metric_file_path = os.path.join(path, "metric.csv")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                start_logits = []
                end_logits = []
                
            running_loss = 0.0
            dataloader = dataloaders[phase] # Class: torch.utils.data.dataloader.DataLoade
            dataset_sizes = len(dataloader.dataset)
            # Iterate over data.
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    start_logit, end_logit, loss = model(batch) #**model(**batch) for other model
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                    else:
                        start_logits.append(start_logit.cpu().numpy())
                        end_logits.append(end_logit.cpu().numpy())
                running_loss += loss.item()
                
            epoch_loss = running_loss / dataset_sizes                     
            if phase == 'train':
                epoch_loss_train.append(epoch_loss)
                print(f'{phase} Loss: {epoch_loss:.4f}')
            else:
                epoch_loss_val.append(epoch_loss)
                start_logits = np.concatenate(start_logits)
                end_logits = np.concatenate(end_logits)
                metric_value = compute_metrics(start_logits, end_logits, raw_dataset_dict, Token_dataset_dict)
                metric_val.append([metric_value['f1'], metric_value['exact_match']])
                print(f'{phase} Loss: {epoch_loss:.4f} Metric: {metric_value}')
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss_val:
                best_loss_val = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss}, model_file_path)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # Export training progress
    loss_df = pd.DataFrame({'epoch': range(1, num_epochs+1), 'epoch_loss_train':epoch_loss_train, 'epoch_loss_val':epoch_loss_val})
    metric_df =  pd.DataFrame(metric_val)
    metric_df.columns = ['F1', 'EM']
    result_df = pd.concat([loss_df, metric_df], axis=1)
    result_df.to_csv(metric_file_path, index=False)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model