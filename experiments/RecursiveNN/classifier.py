# -*- coding: utf-8 -*-
import sys
import time

import numpy as np
import pandas as pd
from random import shuffle

import matplotlib.pyplot as plt
import itertools

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm

import tree_class as tr
from utils import Vocab,  clean_text_simple, load_embedding, arrange_embeddings

class RNN_Model(nn.Module):
  # Inspired from https://github.com/kingtaurus/cs224d/blob/master/assignment3/codebase_release/rnn_pytorch.py  
  def __init__(self, vocab, embed_size=300, label_size=3):
    super(RNN_Model, self).__init__()
    self.embed_size = embed_size
    self.label_size = label_size
    self.vocab = vocab
    self.embedding = nn.Embedding(int(self.vocab.__len__()), self.embed_size)
    self.fcl = nn.Linear(self.embed_size, self.embed_size, bias=True)
    self.fcr = nn.Linear(self.embed_size, self.embed_size, bias=True)
    self.projection = nn.Linear(self.embed_size, self.label_size , bias=True)
    self.activation = F.relu
    self.node_list = []

  def walk_tree(self, node):
    if node.isLeaf:
      word_id = torch.LongTensor([self.vocab.encode(node.word)])
      current_node = self.embedding(Variable(word_id))
      self.node_list.append(self.projection(current_node).unsqueeze(0))
    else:
      left  = self.walk_tree(node.left)
      right = self.walk_tree(node.right)
      current_node = self.activation(self.fcl(left) + self.fcl(right))
      self.node_list.append(self.projection(current_node).unsqueeze(0))
    return current_node
    
  def forward(self, x):
    """
    Forward function accepts input data and returns a Variable of output data
    """
    self.node_list = []
    root_node = self.walk_tree(x.root)
    all_nodes = torch.cat(self.node_list)
    #now I need to project out
    return all_nodes


class Classifier():
    def __init__(self):
        self.batch_size = 32
        self.embed_size = 300
        self.label_size = 3
        self.max_epochs = 30
        self.lr = 0.01
        self.use_pretrained_embeddings = True
        self.vocab = Vocab()

    def fit(self, X, y):
        train_tokens = pd.Series(X).apply(clean_text_simple)

        train_data = tr.loadTrees(train_tokens,y)
        train_sents = [tree.get_words() for tree in train_data]
          
        self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))
        self.model = RNN_Model(self.vocab, embed_size=self.embed_size)
        
        if self.use_pretrained_embeddings: 
            # get vectors in the right order according to the vocab
            raw_embedding = load_embedding(
            filename='experiments/keras_glove/glove.6B/glove.6B.300d.txt')
            embedding_vectors = arrange_embeddings(raw_embedding, self.vocab)
            self.model.embedding.weight.data = torch.Tensor(embedding_vectors)
        
        train_loss_history = []
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, dampening=0.0)
        training_start_time = time.time()
        for epoch in range(self.max_epochs):
            print("epoch = ", epoch)
            shuffle(train_data)
            if (epoch % 10 == 0) and epoch > 0:
                for param_group in optimizer.param_groups:
                    #update learning rate
                    print("Dropping learning from %f to %f"%(param_group['lr'], 0.5 * param_group['lr']))
                    param_group['lr'] = 0.5 * param_group['lr']
            for step, tree in enumerate(train_data):
        
                # Perform batch training
                if step % self.batch_size == 0 and step > 0:
                    # zero the parameter gradients
                    optimizer.zero_grad()
              
                all_nodes_prediction = self.model(tree)
                root_prediction = all_nodes_prediction[-1]
                
                root_label = int(tree.root.label)
                torch_label = torch.LongTensor([root_label])
              
                objective_loss = F.cross_entropy(input=root_prediction, target=Variable(torch_label))
              
                train_loss_history.append(objective_loss.data[0])
                if step % 50 == 0 and step > 0:
                  print("step %3d, last loss %0.3f, mean loss (%d steps) %0.3f" % (step, objective_loss.data[0], 700, np.mean(train_loss_history[-700:])))
             
                if objective_loss.data[0] > 5 and epoch > 10:
                    #interested in phrase that have large loss (i.e. incorrectly classified)
                    print(' '.join(tree.get_words()))
                
                if np.isnan(objective_loss.data[0]):
                    print("object_loss was not a number")
                    sys.exit(1)
                else:
                    objective_loss.backward()
                    clip_grad_norm(self.model.parameters(), 5, norm_type=2.)
                    # Perform batch training
                    if step % self.batch_size == 0 and step > 0:
                        optimizer.step()
            print("Epoch done, took {:.2f}s".format(time.time() - training_start_time))
            model_path = "recursiveNN_model_epoch{}.pth".format(epoch)
            torch.save(self.model.state_dict(), model_path)
        plt.figure()
        plt.plot(train_loss_history)
        plt.show()
        print("Training finished.")

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y = np.argmax(y_proba, axis=1)
        return y

    def predict_proba(self, X):
        y = np.empty((0,3))
        tokens = pd.Series(X).apply(clean_text_simple)
        trees_data = [tr.Tree(tokens[i],None) for i in range(tokens.shape[0])]
        for step, tree in enumerate(trees_data): 
            all_nodes_prediction = self.model(tree)
            root_prediction = all_nodes_prediction[-1]
            y = np.append(y, root_prediction.data.numpy(), axis=0)
        return y
