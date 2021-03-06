#!/usr/bin/python

import bz2,contextlib
import numpy as np
import sys
import json
import cPickle as pickle
#import simplejson as json
from constants import *
from common.util import Alphabet,ETag,ConstTag
import importlib
from collections import defaultdict

_FEATURE_TEMPLATES_FILE = './feature/basic_feats.templates'

class Model():
    """weights and templates"""
    #weight = None
    #n_class = None
    indent = " "*4
    #feature_codebook = None
    #class_codebook = None

    def __init__(self,errlog=sys.stdout):
        self.errlog = errlog
        self.weight = None
        self.aux_weight = None
        self.avg_weight = None # for store the averaged weights
        #self._feature_templates_list = []
        #self._feats_gen_filename = None
        self.feats_generator = None
        self.class_codebook = Alphabet()
        self.feature_codebook = Alphabet()
        self.tagdict = {}
        self.counts = None

    def setup(self,instances):
        raise NotImplementedError("Must implement 'setup' method")

    def _set_class_weight(self,n_class,init_feature_dim):
        raise NotImplementedError("Must implement method")
        
    def save_model(self,model_filename):
        #pickle.dump(self,open(model_filename,'wb'),pickle.HIGHEST_PROTOCOL)
        print >> self.errlog, 'Model info:'
        print >> self.errlog,'class size: %s '%(self.class_codebook.size())
        #print >> self.errlog,'feature codebook size: %s' % (self.feature_codebook.size())
        #print 'weight shape: %s' % (self.avg_weight.shape)
        #print >> self.errlog,'weight shape: %s' % (','.join(('%s:%s')%(i,w.shape) for i,w in enumerate(self.avg_weight)))
        #print >> self.errlog,'word-pos table: %s' % (len(self.counts))

        with contextlib.closing(bz2.BZ2File(model_filename, 'wb')) as f:
            pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
        #json.dump(self.toJSON(),open(model_filename,'wb'))
        
    @staticmethod
    def load_model(model_filename):
        with contextlib.closing(bz2.BZ2File(model_filename, 'rb')) as f:
            model = pickle.load(f)
        return model
        #return pickle.load(open(model_filename,'rb'))


class SWTaggerModel(Model):
    
    def setup(self,instances):
        '''Make a tag dictionary for single-tag words. from TextBlob'''
        counts = defaultdict(lambda: defaultdict(int))
        for inst in instances:
            for word, tag in inst:
                counts[word][tag] += 1
                self.class_codebook.add(tag)

        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
            if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
                self.tagdict[word] = tag

        #for inst in instances:
        #    for i, (wd, tag) in enumerate(inst):
        #        self.class_codebook.add(tag)
        self._set_class_weight(self.class_codebook.size())

    def _set_class_weight(self,n_class,init_feature_dim=5*10**5):
        self.weight = {}
        self.avg_weight = {}
        self.aux_weight = {}
        
class TaggerModel(Model):
    '''weight model for POS tagger'''
    def setup(self,instances):
        '''Make a tag dictionary for single-tag words. from TextBlob'''
        self.counts = defaultdict(lambda: defaultdict(int))
        for inst in instances:
            for word, tag in inst:
                self.counts[word][tag] += 1
                self.class_codebook.add(tag)

        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in self.counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
            if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
                self.tagdict[word] = tag

        #for inst in instances:
        #    for i, (wd, tag) in enumerate(inst):
        #        self.class_codebook.add(tag)
        self._set_class_weight(self.class_codebook.size())

    
    def _set_class_weight(self,n_class,init_feature_dim = 5*10**5):
        
        #if n_rel == None:
        #    n_rel = [1]*n_class
        #assert len(n_rel) == n_class

        
        #self.weight = [np.zeros(shape = (init_feature_dim,nt,nr),dtype=WEIGHT_DTYPE) for nr,nt in zip(n_rel,n_tag)]
        #self.aux_weight = [np.zeros(shape = (init_feature_dim,nt,nr),dtype=WEIGHT_DTYPE) for nr,nt in zip(n_rel,n_tag)]
        #self.avg_weight = [np.zeros(shape = (init_feature_dim,nt,nr),dtype=WEIGHT_DTYPE) for nr,nt in zip(n_rel,n_tag)]

        self.weight = [np.zeros(shape = (init_feature_dim),dtype=WEIGHT_DTYPE) for _ in range(n_class)]
        self.aux_weight = [np.zeros(shape = (init_feature_dim),dtype=WEIGHT_DTYPE) for _ in range(n_class)]
        self.avg_weight = [np.zeros(shape = (init_feature_dim),dtype=WEIGHT_DTYPE) for _ in range(n_class)]



    