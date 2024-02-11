import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor, AlignProcessor
from sklearn.model_selection import train_test_split

from utils import preprocessing_text
from config import (humanitarian_train, humanitarian_dev, humanitarian_test,
                    informative_train, informative_dev, informative_test, data_dir, crisismmd, pretrained_models)
                    

class CustomDataset(Dataset):
    def __init__(self,
                 dataframe,
                 image_dir,
                 processor, 
                 classification_task,
                 device):
        '''
        Description: Creates a custom dataset in pytorch. Returns processed output from given processor.
        
        Argument:
        ---------
        dataframe: (pd.DataFrame), Dataframe where image file names, texts, labels are present
        image_dir: (str) Directory where images are located
        processor: Processor to process both text and image modalities.
        classification_task: (str) Informative or Humanitarian.
        device: (str) CUDA device (eg. cuda:0) or CPU (not tested on CPU or mps)
        '''
        self.data = dataframe
        self.img_dir = image_dir
        self.processor = processor
        self.task = classification_task
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        # read image
        image = Image.open(img_path).convert("RGB")
        # read text
        text = row['tweet_text']
        # processsor to process image and text
        process = self.processor(
            text=text,
            images=image,
            return_tensors='pt',
            padding="max_length").to(self.device)

        processed_data = {key: val.squeeze() for key, val in process.items()}

        # classes from string to integer
        if self.task == 'Informative':
            classes = {'informative': 0,
                        'not_informative': 1}

        elif self.task == 'Humanitarian':
            classes = {'affected_individuals': 0,
                        'rescue_volunteering_or_donation_effort': 1,
                        'infrastructure_and_utility_damage': 2,
                        'other_relevant_information': 3,
                        'not_humanitarian': 4}

        else:
            raise ValueError('Task not recognized')

        # labels in integer
        label = classes[row['label']]
        return processed_data, label


def preprocess_data(df, processor_name):
    # tweets preprocessing
    df['tweet_text'] = df['tweet_text'].apply(preprocessing_text, dataset='CrisisMMD')
    # truncate to match context length
    if processor_name.startswith('CLIP'):
        df['tweet_text'] = df['tweet_text'].apply(lambda x: x[:110])
    return df


# get data for finetuning
def get_finetuning_data(task,
                        batch_size,
                        device,
                        eval_finetune,
                        model_name):

    # instantiate processor for CLIP or Align model
    if model_name.startswith('CLIP'):
        processor = CLIPProcessor.from_pretrained(pretrained_models['CLIP'])
    elif model_name.startswith('ALIGN'):
        processor = AlignProcessor.from_pretrained(pretrained_models['ALIGN'])
    else:
        raise NotImplementedError('CLIP and ALIGN processors only implemented')

    # if true, returns only the test dataloader
    if eval_finetune:
        test_file = informative_test if task == 'Informative' else humanitarian_test
        test_data = pd.read_csv(os.path.join(data_dir, crisismmd, test_file),
                                sep='\t', 
                                usecols=['tweet_text', 'image', 'label'])

        # preprocess test data
        test_data = preprocess_data(df=test_data,
                                    processor_name=model_name)

        # custom dataset
        test_data = CustomDataset(dataframe=test_data,
                                  image_dir=data_dir,
                                  processor=processor,
                                  classification_task=task,
                                  labels=True,
                                  device=device)
        
        # test dataloader
        test_dataloader = torch.utils.data.DataLoader(test_data,
                                                      batch_size=batch_size,
                                                      shuffle=False)
        return test_dataloader
        
    # if eval_finetune false then returns train and dev dataloader
    train_file = informative_train if task == 'Informative' else humanitarian_train
    dev_file = informative_dev if task == 'Informative' else humanitarian_dev

    train_data = pd.read_csv(os.path.join(data_dir, crisismmd, train_file),
                             sep='\t', usecols=['tweet_text', 'image', 'label'])

    dev_data = pd.read_csv(os.path.join(data_dir, crisismmd, dev_file),
                           sep='\t', usecols=['tweet_text', 'image', 'label'])

    # preprocess data
    train_data = preprocess_data(df=train_data,
                                 processor_name=model_name)
    dev_data = preprocess_data(dev_data,
                               processor_name=model_name)

    # custom dataset
    train_data = CustomDataset(dataframe=train_data,
                               image_dir=data_dir,
                               processor=processor,
                               classification_task=task,
                               labels=True,
                               device=device)

    dev_data = CustomDataset(dataframe=dev_data,
                             image_dir=data_dir,
                             processor=processor,
                             classification_task=task,
                             labels=True,
                             device=device)

    # dataloader
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)

    dev_dataloader = torch.utils.data.DataLoader(dev_data,
                                                 batch_size=batch_size,
                                                 shuffle=False)
    
    return train_dataloader, dev_dataloader
