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
    data_dir = args.data_directory
    checkpoint_path = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    epochs = args.epochs
    hidden_units = args.hidden_units
    output_size = args.output
    gpu = args.gpu
  

    # load and process images from data directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    trainloader, validloader, testloader, class_to_idx = process_data(train_dir, valid_dir, test_dir)
    

    # load mapping for pretrained feature network and corresponding input size
    arch_dict = {'alexnet': 'alexnet', 'densenet': 'densenet121', 'resnet': 'resnet18', 'vgg': 'vgg16'}
    input_size_dict = {'alexnet': 9216, 'densenet': 1024, 'resnet': 512, 'vgg': 25088}
    

    # configure feature network
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)  
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
    elif arch == 'resnet':
        model = models.resnet18(pretrained=True)
    elif arch == 'vgg':
        model = models.vgg16(pretrained=True)
    else:
        print('Try again. Enter one of these models: alexnet, densenet, resnet, vgg.')

    
    # freeze feature model parameters
    for param in model.parameters():
        param.requires_grad = False


    # define classifier network  
    input_size = input_size_dict[arch]
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(hidden_units, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    

    # attach project speciic classifier 
    model.classifier = classifier

    # define loss
    criterion = nn.NLLLoss()

    # Only train classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    # Train model
    model, optimizer = train(model, trainloader, validloader, criterion, optimizer, epochs, gpu)    

    # Test Accuracy
    calc_accuracy(model, criterion, testloader, gpu)
    
    # Save Checkpoint
    model.class_to_idx = class_to_idx
    model.arch = arch
    model.learning_rate = learning_rate
    save_model(model, optimizer, checkpoint_path)


# function definitions

def get_args():
    ''' Get arguments from command line.'''
    parser = argparse.ArgumentParser()  
    parser.add_argument('data_directory', type=str, default= 'flowers', 
                        help= 'location for training and validation data')
    parser.add_argument('--save_dir', type=str, default='ckpt_flowers.pth', 
                        help= 'location of model checkpoint')
    parser.add_argument('--arch', type=str, default="densenet",
                        help= 'pre-trained models: alexnet, densenet, resnet, vgg')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate used in model training')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs in model training')
    parser.add_argument('--hidden_units', type=int, default=100,
                        help="number of hidden units in classifier")
    parser.add_argument('--gpu', type=bool, default = True,
                        help="use GPU or CPU for model training. True = GPU, False = CPU")
    parser.add_argument('--output', type=int, default=102,
                        help="number of categories for model prediction")    
    return parser.parse_args()


def process_data(train_dir, valid_dir, test_dir):
    '''Load and transform data so model can successfully train.'''
    # define rules for image transformations 
    train_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(35),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.458,0.456,0.406],
                                                                [0.229,0.224,0.225])])


    valid_transforms = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.458,0.456,0.406],
                                                                    [0.229,0.224,0.225])])


    test_transforms = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.458,0.456,0.406],
                                                                    [0.229,0.224,0.225])])
    
    # Load data with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)


    # Using the image datasets and the trainforms, define dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 64)
    print('Data successfully loaded.')
    return trainloader, validloader, testloader, train_datasets.class_to_idx



def get_device(gpu):  
    '''Turn on GPU or CPU option based on environment. '''
    if gpu is True:
        device = torch.device('cuda')
    else: 
        device = torch.device('cpu')      
    return device


def train(model, trainloader, validloader, criterion, optimizer, epochs, gpu):  
    '''Train model to optimize classifier parameters.'''
    # turn on GPU or CPU per environment
    device = get_device(gpu)
    model.to(device)

    # record trainig time
    since = time.time()
    print('Training starts ...')

    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the designated device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()   # no dropout
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}   "
                      f"Train loss: {running_loss/print_every:.3f}   "
                      f"Validation loss: {valid_loss/len(validloader):.3f}   "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    return model, optimizer
    
    time_elapsed = time.time() - since
    print('Training completes in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def calc_accuracy(model, criterion, testloader, gpu):
    '''Calculate prediction accuracy on test dataset.'''
    model.eval()   # no dropout
    
    # turn on GPU or CPU per environment
    device = get_device(gpu)
    model.to(device)
    
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(testloader):.3f}")
    model.train()          

 
def save_model(model, optimizer, checkpoint_path):
    '''Save model parameters as checkpoint to file.'''   
    checkpoint = {'arch': model.arch,
                  'classifier' : model.classifier,
                  'model_state_dict' : model.state_dict(), 
                  'optimizer_state_dict' : optimizer.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                  'learning_rate' : model.learning_rate
                  }
    torch.save(checkpoint, checkpoint_path)
    print('Model successfully checkpointed.')


# run program
if __name__ == "__main__":
    main()