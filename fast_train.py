#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:24:00 2019

@author: lichengcheng
"""
import torch.nn as nn
import numpy as np 
import time

class RBM(nn.Module):
    '''
    This class defines all the functions needed for an BinaryRBM model
    where the visible and hidden units are both considered binary
    '''

    def __init__(self, visible_units, hidden_units, lambd, batch_size, eta,threshold=0.1):
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.lr = eta
        self.eta=eta 
        self.batch_size = batch_size
        self.threshold=threshold #a small threshold of 0.1 to further remove the noise
        self.lambd = lambd #penalty term

        #initialization
        self.W = np.random.uniform(0,2.5,(self.visible_units,self.hidden_units))#weight matrix # weights
        self.h_bias = np.random.normal(0,0.1,self.hidden_units) #hidden layer bias
        self.v_bias = np.random.uniform(-0.5,-2, self.visible_units) #visible layer bias

    def sampling(self, p): #sampling binomial
        return np.random.binomial(1, p)
    
    def sigmoid(self,x): #sigmoid function
        return 1 / (1 + np.exp(-x))

    def visible_to_hidden(self, v): #sampling hidden units and probability
        n=np.shape(v)[0]
        p_h = np.matmul(v, self.W)+np.outer(np.ones(n),self.h_bias)#Compute p(h|v)
        p_h=self.sigmoid(p_h)
        sample_h = self.sampling(p_h) #sample an array of hidden units
        return p_h, sample_h


    def hidden_to_visible(self, h): #sampling visible units and probability
        n=np.shape(h)[0]
        p_v = np.matmul(h,self.W.transpose())+np.outer(np.ones(n), self.v_bias) #compute p(v|h)
        p_v=self.sigmoid(p_v)
        sample_v = self.sampling(p_v) #sample an array of visible units
        return p_v, sample_v
    
    def thresholdingW(self, W, threshold=0.1): #convert W to Q
        W=np.absolute(W)
        W[W<threshold]=0
        W[W>=threshold]=1
        W=np.array(W)
        return W


    def contrastive_divergence(self, input,q,u, training = True, n_gibbs_steps = 1): #CD-1 algorithm
        # positive phase
        positive_hidden_probs, positive_hidden_samples = self.visible_to_hidden(input)
        # calculate W via positive phase
        positive_associations = np.matmul(input.transpose(), positive_hidden_probs)

        # negative phase
        hidden_activations = positive_hidden_samples
        for i in range(n_gibbs_steps):
            visible_probs, visible_samples = self.hidden_to_visible(hidden_activations)
            hidden_probs, hidden_activations = self.visible_to_hidden(visible_probs)

        # calculate W via negative side
        negative_associations = np.matmul(visible_samples.transpose(), hidden_probs)

        # update parameters
        if training: 
            batch_size = self.batch_size
            #L1 implementation
            q_c_p=np.array(q)
            q_c_n=np.array(q)
            w_update = self.W+self.lr*(positive_associations-negative_associations)/batch_size
            w_c_p=np.array(w_update)
            w_c_n=np.array(w_update)
            ind_p=np.nonzero(w_c_p>0)
            ind_n=np.nonzero(w_c_p<0)
            w_c_p[w_c_p<=0]=0
            w_c_n[w_c_n>=0]=0
            q_c_p[ind_n]=0
            q_c_n[ind_p]=0
            w_c_p=w_c_p-q_c_p-u
            w_c_p[w_c_p<0]=0
            w_c_n=w_c_n-q_c_n+u
            w_c_n[w_c_n>0]=0
            w_update1=w_c_p+w_c_n
            v_bias_update = np.mean(input - visible_samples, axis=0)
            h_bias_update = np.mean(positive_hidden_samples - hidden_activations, axis=0)

            self.W = w_update1
            self.h_bias += self.lr * h_bias_update
            self.v_bias += self.lr * v_bias_update

        # compute reconstruction error
        error = np.mean(np.sum((input - visible_samples)**2, axis = 0))

        return (error, w_update)

    def train(self, data, num_epochs = 50):
        start = time.time()
        N=np.shape(data)[0]
        error=[]
        tim=[]
        u=0
        step=0
        q=np.zeros((self.visible_units, self.hidden_units))
        for epoch in range(num_epochs):
            totalCost=0
            batch_size = self.batch_size
            data_batch = data[:batch_size]
            i = 0
            while data_batch.shape[0] > 0:
                u+=self.lambd*self.lr
                cost, w_half = self.contrastive_divergence(data_batch,q,u)
                totalCost+=cost
                q+=self.W-w_half
                i += 1
                step+=1
                self.lr=self.eta/(1+step/self.batch_size) #linearly decreasing learning rate
                data_batch = data[i*batch_size:(i+1)*batch_size]        
            end1= time.time()
            currCost=totalCost/(N/batch_size) #MSE scaled up for the batch_size
            error.append(currCost)
            tim.append(end1-start) #time for each epoch
        Q=self.thresholdingW(W=self.W, threshold=self.threshold)
        self.W[np.abs(self.W)<0.1]=0
        return [error, tim, currCost,Q, self.W]
