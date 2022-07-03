import argparse
import sys

import torch
from torch import nn, optim
import numpy as np
from data import mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()
        images = train_set[0]
        labels = train_set[1]
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        epochs = 8

        train_loss = []
        for e in range(epochs):
            #running_loss = 0
            
        
            optimizer.zero_grad()
        
            log_ps = model(images)
            
            loss = criterion(log_ps,labels)
            loss.backward()
            optimizer.step()
            print(e)
        
        
            train_loss.append(loss.item())
            
        torch.save(model, 'trained_model.pt')
            
        #plt.plot(range(epochs),train_loss)
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        _, test_set = mnist()
        images = test_set[0]
        labels = test_set[1]
        
        with torch.no_grad():
            # validation pass here
            
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            #print(np.shape(labels.reshape(5000,1)))
            #print(np.shape(top_class))
            #equals = (top_class == labels.reshape(5000,1))
            equals = top_class == labels.view(*top_class.shape)
            #print(np.shape(equals))
            #print(type(labels))
            #equals = torch.tensor(equals)
            #accuracy = torch.mean(torch.FloatTensor(equals))
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            
            
        print(accuracy)

if __name__ == '__main__':
    TrainOREvaluate()
    

    
    
    
    
    
    
    
    
    