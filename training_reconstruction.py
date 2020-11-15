import torch.nn as nn
import numpy as np 

class RBM_train_recon(nn.Module):#debias non-zero entries of W without any penalty

    def __init__(self, visible_units, hidden_units, batch_size, W, Q, eta=2):
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.lr = eta
        self.eta=eta
        self.batch_size = batch_size
        self.W=np.abs(W)#+np.random.normal(0,1.5,(self.visible_units,self.hidden_units))
        #self.W=np.random.normal(0,0.1,(self.visible_units, self.hidden_units))
        self.Q=Q 

        # initialization
        self.h_bias = np.random.normal(0,0.1,self.hidden_units) # hidden layer bias
        self.v_bias =  np.random.uniform(-0.5,-2, self.visible_units)
        #self.v_bias=v_bias#+np.random.normal(0,0.1,visible_units)

    def sampling(self, p):
        return np.random.binomial(1, p)
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def visible_to_hidden(self, v):
        n=np.shape(v)[0]
        p_h = np.matmul(v, self.W)+np.outer(np.ones(n),self.h_bias)# compute p(h|v)
        p_h=self.sigmoid(p_h)
        sample_h = self.sampling(p_h) #sample an array of hidden units
        return p_h, sample_h


    def hidden_to_visible(self, h):
        n=np.shape(h)[0]
        p_v = np.matmul(h,self.W.transpose())+np.outer(np.ones(n), self.v_bias) #compute p(v|h)
        p_v=self.sigmoid(p_v)
        sample_v = self.sampling(p_v) #sample an array of visible units
        return p_v, sample_v

    def contrastive_divergence(self, input, training = True, n_gibbs_steps = 1):

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
            w_update = (positive_associations-negative_associations)/batch_size
            v_bias_update = np.mean(input - visible_samples, axis=0)
            h_bias_update = np.mean(positive_hidden_samples - hidden_activations, axis=0)

            self.W+= self.lr*w_update 
            self.W[self.Q==0]=0 #only update the non-zero parameters
            self.h_bias += self.lr * h_bias_update
            self.v_bias += self.lr * v_bias_update

        # compute reconstruction error
        error = np.mean(np.sum((input - visible_samples)**2, axis = 0))

        return error

    def train(self, data, num_epochs = 100):
        N=np.shape(data)[0]
        error=[]
        step=0
        for epoch in range(num_epochs):
            totalCost=0
            batch_size = self.batch_size
            data_batch = data[:batch_size]
            i = 0
            while data_batch.shape[0] > 0:                
                cost= self.contrastive_divergence(data_batch)
                totalCost+=cost
                i += 1
                step+=1
                self.lr=self.eta/(1+step/self.batch_size)
                data_batch = data[i*batch_size:(i+1)*batch_size]        

            currCost=totalCost/(N/batch_size) #MSE scaled up for the batch_size
            error.append(currCost)
        return self.W, self.h_bias, error.pop()
