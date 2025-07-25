import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, get_file
from keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout

# small library for seeing the progress of loops.
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()

# Loading a text file into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
# get all imgs with their captions
def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions = {}
    for caption in captions[::-1]:
        img, caption = caption.split('\t')
        if img[::-2] not in descriptions:
            descriptions[img[::-2]] = [ caption ]
        else:
            descriptions[img[::-2]].append(caption)
    return descriptions

#Data cleaning- lower casing, removing puntuations and words containing numbers
def cleaning_text(captions):
    table = str.maketrans('','',string.punctuation)
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption.replace('-',' ')
            # splitting the caption into words
            desc = img_caption.split()
            # converts all tokens to lower case
            desc = [word.lower()for word in img_caption]
            # removing punctuation from each token
            desc = [word.translate(table) for word in desc]
            #remove hanging 's and a
            desc = [word for word in desc if len(word)>1]
            #remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            #convert back to string
            img_caption = ' '.join(desc)
            #update the caption
            captions[img][i] = img_caption

    return captions

# build vocabulary of all unique words
def text_vocabulary(descriptions):
    vocab = set()
    for key in descriptions.key():
        [vocab.update(d.aplit())for d in descriptions[key]]

    return vocab
# Save all descriptions in one file 
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    data = '\n'.join(lines)
    file = open(filename,'w')
    file.write(data)
    file.close()

# Set these path according to project folder in you system
dataset_text = "/Users/sreemanti/Documents/youtube/youtube-teach/image caption generator/Flickr8k_text"
dataset_images = "/Users/sreemanti/Documents/youtube/youtube-teach/image caption generator/Flicker8k_Dataset"

# we prepare our text data
filename = dataset_text + "/" + "Flickr8k.token.txt"
#loading the file that contains all data
#mapping them into descriptions dictionary img to 5 captions
descriptions = all_img_captions(filename)
print("Length of descriptions =" ,len(descriptions))

# #cleaning the descriptions
clean_descriptions = cleaning_text(descriptions)

# #building vocabulary 
vocabulary = text_vocabulary(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))

# #saving each description to file 
save_descriptions(clean_descriptions, "descriptions.txt")

def download_with_retry(url, filename, max_retries=3):
    for attempt in range(max_retries):
        try:
            return get_file(filename, url)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Download attempt {attempt + 1} failed. Retrying in 5 seconds...")
            time.sleep(5)
# Replace the Xception model initialization with:           
weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_path = download_with_retry(weights_url,"xception_weights.h5")
# Load the Xception model
model = Xception(include_top=False, pooling='avg', weights = weights_path,)

def extract_features(directory):
    features = {}
    valid_images = [".jpg", ".jpeg", ".png"]
    # Loop through all files in the directory
    for img in tqdm(os.listdir(directory)):
        if os.path.splitext(img)[1].lower() not in valid_images:
            continue
        filename = directory + "/" + img
        image = Image.open(filename)
        image = image.resize((299, 299))
        image = np.expand_dims(image,axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        features[img] = feature

    return features
        
# 2048 feature vector
features = extract_features(dataset_images)
# dump the features to a file
dump(features,open("features.p","wb"))
# load the features from the file
features = load(open("features.p","rb"))

#load the data 
def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    photos_present = [photo for photo in photos if os.path.exists(os.path.join(dataset_images, photo))]
    return photos_present

def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}

    for line in flile.split("\n"):
        word = line.split()
        if len(word)<2:
            continue
        image,image_caption = word[0], word[1:]

        for image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)
        
        return descriptions
    
def load_features(photos):
    #loading all features
    all_features = load(open("features.p","rb"))
    #selecting only needed features
    features = {k:all_features[k] for k in photos}
    return features

filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"

#train = loading_data(filename)
train_images = load_photos(filename)
train_descriptions = load_clean_descriptions("descriptions.txt", train_images)
train_features = load_features(train_images)

def dict_to_list(descriptions):
    all_desc =[]
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    
    return all_desc

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer