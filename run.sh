#!/bin/bash

source virtualenvs/transformers/bin/activate
export PYTHONPATH=$PYTHONPATH:./multimodal_classification

# testing
# python multimodal_classification/additional_experiments/ablation_studies.py --classification_task Humanitarian --device cuda:1 --log_file ablation.log

python multimodal_classification/additional_experiments/mm-claims_experiments.py --single_modality --text_embed --pretrained_model CLIP --epochs 30 --device cuda:2 >> clip_text_result_mm_claims.txt