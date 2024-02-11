import os
import torch
import argparse
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from train import CustomTrain
from utils import parameters_with_layerwise_lr, get_logger
from eval import evaluate_pretraining, evaluate_finetuning
from src.process_crisismmd_data import get_finetuning_data
from src.process_dmd_data import get_dmd_finetuning_data
from model import ClassifyCLIP, ClassifyAlign
from config import data_dir, project_dir, finetuned_path


def get_args():

    parser = argparse.ArgumentParser(description='Multimodal Crisis Classification')

    parser.add_argument('--classification_task', type=str, default=None)
    parser.add_argument('--single_modality', action='store_true')
    # only one among the two should be true, that too only if single_modality is true
    parser.add_argument('--image_embed', action='store_true')
    parser.add_argument('--text_embed', action='store_true')

    # If true, evlauates finetuning, else training mode is on
    parser.add_argument('--eval_mode', action='store_true')

    # if eval_mode is false, training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)

    # even if eval_mode is false, you need batch size
    parser.add_argument('--batch_size', type=int, default=None)

    # name of the pretrained model: It should be set to 'CLIP' or 'ALIGN'
    parser.add_argument('--pretrained_model', type=str, default=None)

    # device: If cuda is available mention the cuda (with device id) (if you 
    # have multiple gpu's) else cpu. Note that distributed training is not 
    # supported as of now.
    parser.add_argument('--device', type=str, default=None)

    # logs file and tensorboard summary writer
    parser.add_argument('--log_file', type=str, default=None)
    # should only required when eval_mode is true
    parser.add_argument('--summary_file', type=str, default=None)

    args = parser.parse_args()
    return args


# driver function
if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device)
    logger = get_logger(args.log_file) if args.log_file else None
    writer = SummaryWriter(args.summary_file) if args.summary_file else None

    num_classes = 2 if args.classification_task == 'Informative' else 5

    # create model architecture for finetuning
    if args.pretrained_model.startswith('CLIP'):
        model = ClassifyCLIP(pretrained_model=args.pretrained_model,
                            num_classes=num_classes,
                            single_modality=args.single_modality,
                            text_embed=args.text_embed,
                            image_embed=args.image_embed)
    elif args.pretrained_model.startswith('ALIGN'):
        model = ClassifyAlign(pretrained_model=args.pretrained_model,
                            num_classes=num_classes,
                            single_modality=args.single_modality,
                            text_embed=args.text_embed,
                            image_embed=args.image_embed)
    else:
        raise NotImplementedError('Only CLIP and ALIGN models implemented')

    # model name under which finetuned model will be saved or loaded (in evaluation mode)
    model_name = f'{args.classification_task}_{args.pretrained_model}'

    if args.single_modality:
        model_name = f'{model_name}_Image' if args.image_embed else f'{model_name}_Text'

    # model name with path (to save or load)
    model_name_path = os.path.join(project_dir, finetuned_path, model_name)

    # training mode
    if not args.eval_mode:
        # get training and dev data for finetuning
        train_data, dev_data = get_finetuning_data(task=args.classification_task,
                                                    batch_size=args.batch_size,
                                                    device=device,
                                                    eval_finetune=args.eval_mode,
                                                    model_name=args.pretrained_model)
        
        # send model to device
        model.to(device)
        
        # instantaite cross entropy loss
        criterion = nn.CrossEntropyLoss()

        # Freeze all layers except the last FC layer for classification
        for idx, (name, param) in enumerate(model.named_parameters()):
            if name != 'fc.weight' and name != 'fc.bias':
                param.requires_grad = False

        # instantiate optimizer
        optimizer = Adam(filter(lambda p: p.requires_grad, 
                                model.parameters()),lr=args.learning_rate)
        
        # log training beginning
        training_info = f'bs{args.batch_size}_lr{args.learning_rate}'
        message = f'Finetuning Started: {training_info} with base model: {args.pretrained_model}'
        logger.info(message)

        # instantiate trainer class
        trainer = CustomTrain(model=model,
                                train_dataloader=train_data,
                                dev_dataloader=dev_data,
                                num_epochs=args.epochs,
                                save_model_file_name=model_name_path,
                                pretrain=args.pretraining,
                                criterion=criterion,
                                optimizer=optimizer,
                                writer=writer,
                                logger=logger,
                                device=device)
        
        # train model (saves best model as well)
        model = trainer.train()

    # evaluate finetuning
    else:

        # get test data for evaluating the classification task
        test_data = get_finetuning_data(task=args.classification_task,
                                        batch_size=args.batch_size,
                                        device=device,
                                        eval_finetune=args.eval_mode,
                                        model_name=args.pretrained_model)
        
        # load model weights to the architecture to evaluate model
        model.load_state_dict(torch.load(model_name_path))
        model.to(device)

        # evaluate saved model
        evaluate_finetuning(model=model,
                            task=args.classification_task,
                            finetuned_model_name=model_name,
                            logger=logger,
                            test_dataloader=test_data)
                                       