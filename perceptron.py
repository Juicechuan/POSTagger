#!/usr/bin/python

import numpy as np
from constants import WEIGHT_DTYPE
from collections import defaultdict


class BasePerceptron():
    '''base class for various average perceptron'''
    def __init__(self,model):
        self.model = model
        self.wstep = 1
        self.num_updates = 0
        #self.reshape_rate = reshape_rate 

    def get_num_updates(self):
        return self.num_updates
                
    def no_update(self):
        self.wstep += 1

    def predict(self):
        raise NotImplementedError("Must implement the 'predict' method")

    #def reshape_weight(self):
    #    raise NotImplementedError("Must implement the 'reshape_weight' method")

    def update_weight(self,class_g,class_b,feats):
        raise NotImplementedError("Must implement the 'update_weight' method")

    def average_weight(self):
        raise NotImplementedError("Must implement the 'average_weight' method")


class SWPerceptron(BasePerceptron):

    def predict(self,feats,train=True):
        scores = defaultdict(float)
        for f in feats:
            if f not in self.model.weight:
                continue
            weights = self.model.weight[f] if train else self.model.avg_weight[f]
            for label, weight in weights.items():
                scores[label] += weight
        return max(self.model.class_codebook.labels(),key=lambda label: (scores[label],label))

    def update_weight(self,class_g,class_b,feats):
        
        for f in feats:
            weights = self.model.weight.setdefault(f,{})
            w_g = weights.get(class_g,0.0)
            w_b = weights.get(class_b,0.0)
            self.model.weight[f][class_g] = w_g + 1.0
            self.model.weight[f][class_b] = w_b - 1.0

            aux_weights = self.model.aux_weight.setdefault(f,{})
            aw_g = aux_weights.get(class_g,0.0)
            aw_b = aux_weights.get(class_b,0.0)
            self.model.aux_weight[f][class_g] = aw_g + float(self.wstep)
            self.model.aux_weight[f][class_b] = aw_b - float(self.wstep)

            avg_weights = self.model.avg_weight.setdefault(f,{})
            avgw_g = avg_weights.get(class_g,0.0)
            avgw_b = avg_weights.get(class_b,0.0)
            
 
        self.num_updates += 1
        self.wstep += 1
        
    def average_weight(self):
        for f in self.model.weight:
            weights = self.model.weight[f]
            aux_weights = self.model.aux_weight[f]
            for label in weights:
                w = weights[label]
                aw = aux_weights[label]
                avgw = round(w - aw/float(self.wstep),3)
                if avgw:
                    self.model.avg_weight[f][label] = avgw
                
    
class Perceptron(BasePerceptron):
    
    #model = None
    #num_updates = 0
    #wstep = 1

    def predict(self,feats,validTagset,train=True):
        feats_indices = map(self.model.feature_codebook.get_index,feats)
        if train:
        # only compute the valid tag set
            weights = [self.model.weight[self.model.class_codebook.get_index(vt)] for vt in validTagset]
        else:
            weights = [self.model.avg_weight[self.model.class_codebook.get_index(vt)] for vt in validTagset]  
        scores = map(lambda w: np.sum(w[[i for i in feats_indices if i is not None]]),weights)

        best_tag = validTagset[np.argmax(scores)]
        
        return best_tag

    def reshape_weight(self,reshape_rate=10**5):
        for class_idx in self.model.class_codebook.indexes():
            w = self.model.weight[class_idx]
            aw = self.model.aux_weight[class_idx]
            avgw = self.model.avg_weight[class_idx]

            self.model.weight[class_idx] = np.vstack((w,np.zeros(shape=(reshape_rate),dtype=WEIGHT_DTYPE)))
            self.model.aux_weight[class_idx] = np.vstack((aw,np.zeros(shape=(reshape_rate),dtype=WEIGHT_DTYPE)))
            self.model.avg_weight[class_idx] = np.vstack((avgw,np.zeros(shape=(reshape_rate),dtype=WEIGHT_DTYPE)))
 
    def update_weight(self,class_g,class_b,feats):
        self.num_updates += 1
        
        class_g_idx = self.model.class_codebook.get_index(class_g)
        class_b_idx = self.model.class_codebook.get_index(class_b)

        #class_l_g = class_l_g if class_l_g else 0
        #class_t_g = class_t_g if class_t_g else 0

        #class_l_b = class_l_b if class_l_b else 0
        #class_t_b = class_t_b if class_t_b else 0
        
        if self.model.weight[class_g_idx].shape[0] <= self.model.feature_codebook.size()+len(feats):
            self.reshape_weight()

        # adding the first seen feature
        for f in feats:
            if not self.model.feature_codebook.has_label(f):
                self.model.feature_codebook.add(f)
            
        feats_indices = map(self.model.feature_codebook.get_index,feats)
        self.model.weight[class_g_idx][feats_indices] += 1
        self.model.aux_weight[class_g_idx][feats_indices] += self.wstep        
        
        #if self.model.weight[class_b_idx].shape[0] <= self.model.feature_codebook[class_b_idx].size()+len(feat_b):
        #    self.reshape_weight(class_b_idx)

        #b_feats_indices = map(self.model.feature_codebook[class_b_idx].get_index,feat_b)
        self.model.weight[class_b_idx][feats_indices] -= 1
        self.model.aux_weight[class_b_idx][feats_indices] -= self.wstep

        self.wstep += 1
        
         
    def average_weight(self):
        for i in self.model.class_codebook.indexes():
            weight = self.model.weight[i]
            aux_weight = self.model.aux_weight[i]
            avg_weight = self.model.avg_weight[i]
            wstep = self.wstep 
            
            #np.divide(aux_weight,wstep+.0,aux_weight)
            np.divide(aux_weight,wstep+.0,avg_weight)
            np.subtract(weight,avg_weight,avg_weight)
        
