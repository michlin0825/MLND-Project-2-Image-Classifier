# import libraries
import argparse

import json
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from PIL import Image

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

def main():
    
    # get arguments from command line
    args = get_args()
    image_path = args.image_path
    checkpoint_path = args.checkpoint
    top_k = args.top_k
    cat_names = args.category_names
    gpu = args.gpu
    
    # get dictionary for class label
    label_map = get_map(cat_names)
                  
    # Get predictions
    top_probs, top_labels, top_flowers = predict(image_path, checkpoint_path, label_map, top_k, gpu)
    
    # Print predictions
    keys = top_flowers
    values = top_probs
    predict_dict = dict(zip(keys, values))
    sorted(predict_dict.items(), key=lambda x: x[1])
    print('Predicted classes and probabilities: ')
    print(predict_dict)
    

# function definitions

def get_args():
    '''Get arguments from command line.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type = str, help = 'path to image for prediction')
    parser.add_argument('checkpoint', type = str, default = 'ckpt_flowers.pth',
                        help = 'checkpoint for model parameters')
    parser.add_argument('--top_k', type=int, default = 5, help="number of classes to predict")
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                        help = 'mapping file to translate label index to label names')
    parser.add_argument('--gpu', type = bool, default = True,
                        help = 'use GPU or CPU for model prediction. True = GPU, False = CPU')
    return parser.parse_args()


def load_model(checkpoint_path):
    '''Load model checkpoint for image prediction.'''
    # Use GPU if it's available 
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)  
    elif checkpoint['arch'] == 'densenet':
        model = models.densenet121(pretrained=True)
    elif checkpoint['arch'] == 'resnet':
        model = models.resnet18(pretrained=True)
    elif checkpoint['arch'] == 'vgg':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError('Base architecture not recognized.')
    
    # freeze feature model parameters 
    for param in model.parameters():
        param.requires_grad = False
    
    # attach checkpoint parameters to classifier 
    model.classifier = checkpoint['classifier']   
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr = checkpoint['learning_rate'])    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    
    model.class_to_idx = checkpoint['class_to_idx']
    print('Checkpoint successfully loaded.')        
    return model  


def process_image(image_path):
    '''Scale, crop, and normalize a PIL image for PyTorch model.'''      
    # Open the image
    from PIL import Image
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    return img


def get_device(gpu):  
    '''Turn on GPU or CPU option based on environment. '''
    if gpu is True:
        device = torch.device('cuda')
    else: 
        device = torch.device('cpu')      
    return device


def get_map(cat_names):
    '''Load class label file.'''
    with open(cat_names, 'r') as f:
        label_map = json.load(f)
    return label_map


def predict(image_path, checkpoint_path, label_map, top_k, gpu):
    '''Predict class from an image file'''       
    # turn on GPU or CPU per environment
    device = get_device(gpu)
      
    # process image
    img = process_image(image_path)
    
    # convert numpy to tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    
    # add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)   
    
    # get model
    model = load_model(checkpoint_path)

    # calculate probs
    model.to(device)
    model_input = model_input.to(device)
    probs = torch.exp(model.forward(model_input))
       
    # Top probs
    # moving tensor back to cpu for display
    probs = probs.to('cpu')
    
    top_probs, top_labs = probs.topk(top_k)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
       
    # convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [label_map[idx_to_class[lab]] for lab in top_labs]
    print('Got image predictions.')
    return top_probs, top_labels, top_flowers

    
# Run program
if __name__ == "__main__":
    main()