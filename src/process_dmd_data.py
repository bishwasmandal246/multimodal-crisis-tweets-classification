import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader
from transformers import CLIPProcessor, AlignProcessor
from sklearn.model_selection import train_test_split
from utils import preprocessing_text
from config import pretrained_models


class DMDDataset(Dataset):
    def __init__(self, root_dir, processor, model_name, device):
        self.root_dir = root_dir
        self.processor = processor
        self.model_name = model_name
        self.device = device
        self.class_names = sorted(os.listdir(root_dir))
        self.class_names = self.class_names[1:]
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            image_dir = os.path.join(class_dir, 'images')
            text_dir = os.path.join(class_dir, 'text')

            image_filenames = sorted(os.listdir(image_dir))
            text_filenames = sorted(os.listdir(text_dir))

            matching_pairs = match_image_text_pairs(image_filenames, text_filenames)

            for img_filename, txt_filename in matching_pairs:
                # just for checking
                if img_filename[:-3] == txt_filename[:-3]:
                    img_path = os.path.join(image_dir, img_filename)
                    txt_path = os.path.join(text_dir, txt_filename)

                    data.append({
                        'image_path': img_path,
                        'text_content': txt_path,
                        'class_label': self.class_names.index(class_name)
                    })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['image_path']
        text_path = self.data[idx]['text_content']
        class_label = self.data[idx]['class_label']

        image = Image.open(img_path).convert('RGB')
        with open(text_path, 'r') as txt_file:
            text_content = txt_file.read()
        
        text = preprocess_data(text_content, self.model_name)

        process = self.processor(
            text=text,
            images=image,
            return_tensors='pt',
            padding="max_length").to(self.device)

        processed_data = {key: val.squeeze() for key, val in process.items()}
        return processed_data, class_label


def match_image_text_pairs(image_filenames, text_filenames):
    image_dict = {img_filename.split('.')[0]: img_filename for img_filename in image_filenames}
    text_dict = {txt_filename.split('.')[0]: txt_filename for txt_filename in text_filenames}

    matching_pairs = []

    for common_name in set(image_dict.keys()) & set(text_dict.keys()):
        matching_pairs.append((image_dict[common_name], text_dict[common_name]))

    return matching_pairs


def preprocess_data(text, processor_name):
    text = preprocessing_text(text)
    if processor_name.startswith('CLIP'):
        text = text[:110]
    # previously the implementation used for ALIGN model had a large context length, but it appears 
    # the model version available now doesn't have a large context length and thus we need to concatenate.
    else:
        text = text[:70]
    return text



# get data for finetuning
def get_dmd_finetuning_data(root_dir,
                            batch_size,
                            device,
                            model_name):

    # instantiate processor for CLIP or Align model
    if model_name.startswith('CLIP'):
        processor = CLIPProcessor.from_pretrained(pretrained_models['CLIP'])
    elif model_name.startswith('ALIGN'):
        processor = AlignProcessor.from_pretrained(pretrained_models['ALIGN'])
    else:
        raise NotImplementedError('CLIP and ALIGN processors only implemented')
    
    dmd_dataset = DMDDataset(root_dir=root_dir,
                             processor=processor,
                             device=device,
                             model_name=model_name)
    
    # Split the dataset indices into train and test indices
    indices = list(range(len(dmd_dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=0.1)

    # Create Subset instances for train and test sets
    train_dataset = Subset(dmd_dataset, train_indices)
    test_dataset = Subset(dmd_dataset, test_indices)

    # Create DataLoader instances for train and test sets
    batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader