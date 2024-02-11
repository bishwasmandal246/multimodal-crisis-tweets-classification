import torch
import torch.nn as nn
from transformers import CLIPModel, AlignModel
from config import pretrained_models


class ClassifyCLIP(nn.Module):
    def __init__(self,
                 pretrained_model,
                 num_classes,
                 single_modality,
                 text_embed,
                 image_embed):
        '''
        Inherits the nn module from torch and wraps the CLIPModel from transformer library to create a model that is used for finetuning to classification problems.

        Arguments:
        ---------
        pretrained_model: Base pretrained model used for finetuning

        num_classes: (int) Number of classes to be classified.

        single_modality: Only use a single modality either text or image. If False, text_embed, image_embed values are ignored.

        text_embed: Only use text embedding from CLIP model.

        image_embed: Only use image embeddings from CLIP model.
        
        NOTE: Do not use both text_embed and image_embed arguments as True.
        '''
        super(ClassifyCLIP, self).__init__()

        # number of hidden units in fc layer. (512 image embeddings + 512 text embeddings)
        hidden_units = 1024

        self.clip_model = CLIPModel.from_pretrained(pretrained_models[pretrained_model])


        if single_modality:

            hidden_units = 512

            # Check both image and text embed are not true
            if image_embed:
                assert text_embed == False
                  
            if text_embed:
                assert image_embed == False
                
        self.fc = nn.Linear(hidden_units, num_classes)
        self.single_modality = single_modality
        self.image_embed = image_embed
        self.text_embed = text_embed
        
        
    def forward(self, **out):
        '''
        Forward method for ClassifyCLIP
        '''
        outputs = self.clip_model(**out)

        if self.single_modality:

            if self.image_embed:
                embeds = outputs.image_embeds
            else:
                embeds = outputs.text_embeds
            
            outputs = self.fc(embeds)
            return outputs
        
        # If single modality is False, then use both the modalities's embeddings
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # concatenate both embeddings
        embeds = torch.cat((image_embeds, text_embeds), dim=1)

        outputs = self.fc(embeds)
        return outputs
    

class ClassifyAlign(nn.Module):
    def __init__(self,
                 pretrained_model,
                 num_classes,
                 single_modality,
                 text_embed,
                 image_embed):
        '''
        Inherits the nn module from torch and wraps the AlignModel from transformer library to create a model that is used for finetuning to classification problems.

        Arguments:
        ---------
        pretrained_model: Base pretrained model used for finetuning

        num_classes: (int) Number of classes to be classified.

        single_modality: Only use a single modality either text or image. If False, text_embed, image_embed values are ignored.

        text_embed: Only use text embedding from Align model.

        image_embed: Only use image embeddings from Align model.
        
        NOTE: Do not use both text_embed and image_embed arguments as True.
        '''
        super(ClassifyAlign, self).__init__()

        # number of hidden units in fc layer. (640 image embeddings + 640 text embeddings)
        hidden_units = 1280

        self.align_model = AlignModel.from_pretrained(pretrained_models[pretrained_model])


        if single_modality:

            hidden_units = 640

            # Check both image and text embed are not true
            if image_embed:
                assert text_embed == False
                  
            if text_embed:
                assert image_embed == False
                
        self.fc = nn.Linear(hidden_units, num_classes)
        self.single_modality = single_modality
        self.image_embed = image_embed
        self.text_embed = text_embed
        
        
    def forward(self, **out):
        '''
        Forward method for ClassifyAlign
        '''
        outputs = self.align_model(**out)

        if self.single_modality:

            if self.image_embed:
                embeds = outputs.image_embeds
            else:
                embeds = outputs.text_embeds
            
            outputs = self.fc(embeds)
            return outputs
        
        # If single modality is False, then use both the modalities's embeddings
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # concatenate both embeddings
        embeds = torch.cat((image_embeds, text_embeds), dim=1)

        outputs = self.fc(embeds)
        return outputs
