# data directory
data_dir = ''

# project directory
project_dir = './multimodal_classification'

# crisismmd folder where train, dev, test data is located
crisismmd = 'crisismmd_datasplit_agreed_label'

# data for humanitrain task in train, dev, test split
humanitarian_train = 'task_humanitarian_text_img_agreed_lab_train.tsv'
humanitarian_dev = 'task_humanitarian_text_img_agreed_lab_dev.tsv'
humanitarian_test = 'task_humanitarian_text_img_agreed_lab_test.tsv'

# data for informative task in train, dev, test split
informative_train = 'task_informative_text_img_agreed_lab_train.tsv'
informative_dev = 'task_informative_text_img_agreed_lab_dev.tsv'
informative_test = 'task_informative_text_img_agreed_lab_test.tsv'

# saved model locations
finetuned_path = 'models'

# pretrained model names
pretrained_models = {'CLIP': 'openai/clip-vit-base-patch32',
                     'ALIGN': 'kakaobrain/align-base'}
