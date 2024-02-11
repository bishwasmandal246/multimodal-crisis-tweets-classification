import re
import inflect
import logging
import unicodedata
import contractions
import numpy as np


# remove mentions from texts
def remove_mentions(x): return re.sub("@[A-Za-z0-9_]+", "", x)

# remove hashtags from texts
def remove_hashtags(x): return re.sub("#[A-Za-z0-9_]+", "", x)

# fix contactions in text
def replace_contractions(x): return contractions.fix(x)

# remove urls from texts
def remove_urls(x): return re.sub(r"http\S+", "", x)

# remove the token '<URL>' from texts
def remove_url_token(x): return x.replace("<URL>", "")


# make all letters or characters in words to lowercase characters
def to_lowercase(words):
    new_words = []
    for word in words.split(" "):
        new_word = word.lower()
        new_words.append(new_word)
    return " ".join(new_words)


# remove any non ascii characters from the texts
def remove_non_ascii(words):
    new_words = []
    for word in words.split(" "):
        new_word = unicodedata.normalize('NFKD',word).encode(
            'ascii','ignore').decode('utf-8','ignore')
        new_words.append(new_word)
    return " ".join(new_words)


# remove punctuations form texts
def remove_punctuation(words):
    new_words = []
    for word in words.split(" "):
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return " ".join(new_words)


# replace numbers to words
def replace_numbers(words):
    p = inflect.engine()
    new_words = []
    for word in words.split(" "):
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return " ".join(new_words)


# wrapper function to wrap all the above mentioned functions into one
def preprocessing_text(words, dataset):
    words = remove_mentions(words)
    if dataset == 'CrisisMMD':
        words = remove_hashtags(words)
    words = remove_urls(words)
    words = remove_url_token(words)
    if dataset == 'CrisisMMD':
        words = replace_contractions(words)
    words = replace_numbers(words)
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = to_lowercase(words)
    return words


# given image and text embeddings, returns the similarity scores
def normalized_cosine_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.detach().cpu().numpy(
    ) @ image_features.detach().cpu().numpy().T
    img_text_similarity = (np.diag(similarity))
    return img_text_similarity


# logger to write results in a file
def get_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    log_format = '%(asctime)s | %(levelname)s: %(message)s'
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    return logger


# varying learning rate in different layers for CLIP/CLIPSurgery/Align model
# Were used for pretraining experiments, but not required for finetuning
def parameters_with_layerwise_lr(model, lr):
    # should be customised as per the model and requirements
    layer_names = []
    for _, (name, param) in enumerate(model.named_parameters()):
        layer_names.append(name)
    # group layers as per requirements
    layer_name_100 = layer_names[:131] + layer_names[197:330]
    layer_name_10 = layer_names[131:179] + layer_names[330:378]
    layer_name_1 = layer_names[179:197] + layer_names[378:]
    parameters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in layer_name_100:
                parameters.append({'params': param, 'lr': lr / 100})
            elif name in layer_name_10:
                parameters.append({'params': param, 'lr': lr / 10})
            elif name in layer_name_1:
                parameters.append({'params': param, 'lr': lr})
            else:
                raise ValueError("All learnable parameters not used.")
    return parameters
