#!/usr/bin/python

from nltk.corpus import brown
from constants import START,END
import numpy as np
import sys, os, random
from model import Model,SWTaggerModel
from perceptron import Perceptron,SWPerceptron
import time,datetime
from nltk.tree import Tree

#wsj_corpus_path = '/home/j/clp/chinese/corpora/wsj/00'
WSJ_CORPUS_PATH = '/Users/Trevor/Projects/Corpus/wsj'

def wsjtree2pos(wsj_corpus_path):
    print >> sys.stderr, "Reading in corpus..."
    sentences = []
    for d in os.listdir(wsj_corpus_path):
        if os.path.isdir(wsj_corpus_path+'/'+d) and d != 'CVS' and int(d) < 8:            
            for f in os.listdir(wsj_corpus_path+'/'+d): 
                if f.endswith('.mrg'):
                    fname = wsj_corpus_path+'/'+d+'/'+f
                    #print fname
                    tree_f = open(fname,'r')
                    tree_string = ''
                    for line in tree_f:
                        if line.strip():
                            if line.startswith('( (') or line.startswith('(('):
                                if tree_string:
                                    tr = Tree(tree_string)
                                    sentences.append(tr.pos())
                                    tree_string = line.strip()
                                else:
                                    tree_string = line.strip()
                            else:
                                tree_string += line.strip()
                    if tree_string:
                        tr = Tree(tree_string)
                        sentences.append(tr.pos())

    return sentences
                        


class POSTagger():

    #model = None
    def __init__(self,model):
        self.model = model
        self.perceptron = SWPerceptron(model)
        self.errlog = sys.stderr

    def tag(self,test_instances,interval=500):
        start_time = time.time()
        prev, prev2 = START
        c = 0.
        n = 0.
        for sent_id,inst in enumerate(test_instances,1):
            context = START + [self._normalize(w) for w,t in inst] + END
            for i,(word,gold_tag) in enumerate(inst):
                best_tag = self.model.tagdict.get(word)
                if not best_tag:
                    feats = self._get_features(i, word, context, prev, prev2)
                    #validTagset = self.model.counts[word].keys()
                    #if not validTagset: validTagset = self.model.class_codebook.labels()
                    best_tag = self.perceptron.predict(feats,train=False)
                prev2 = prev
                prev = best_tag
                c += best_tag == gold_tag
                n += 1.

            if sent_id % interval == 0:
                p = c/n
                print >> self.errlog,"Over "+str(sent_id)+" sentences ","Accuracy:%s" % (p)

        print >> self.errlog,"One pass on %s sentences takes %s" % (str(sent_id),datetime.timedelta(seconds=round(time.time()-start_time,0)))
        pt = c/n
        #r = n_correct_total/n_gold_total
        #f = 2*p*r/(p+r)
        print >> self.errlog,"Total Accuracy: %s" % (pt)
        
    def train(self,train_instances,interval=500):
        start_time = time.time()
        prev, prev2 = START
        c = 0.
        n = 0.
        for sent_id,inst in enumerate(train_instances,1):
            context = START + [self._normalize(w) for w,t in inst] + END
            for i,(word,gold_tag) in enumerate(inst):
                best_tag = self.model.tagdict.get(word)
                if not best_tag:
                    feats = self._get_features(i, word, context, prev, prev2)
                    #validTagset = self.model.counts[word].keys()
                    best_tag = self.perceptron.predict(feats)
                    if best_tag != gold_tag:
                        self.perceptron.update_weight(gold_tag,best_tag,feats)
                    else:
                        self.perceptron.no_update()
                prev2 = prev
                prev = best_tag
                c += best_tag == gold_tag
                n += 1.

            if sent_id % interval == 0:
                p = c/n
                print >> self.errlog,"Over "+str(sent_id)+" sentences ","Accuracy:%s" % (p)

        print >> self.errlog,"One pass on %s sentences takes %s" % (str(sent_id),datetime.timedelta(seconds=round(time.time()-start_time,0)))
        pt = c/n
        #r = n_correct_total/n_gold_total
        #f = 2*p*r/(p+r)
        print >> self.errlog,"Total Accuracy: %s" % (pt)



    def _normalize(self, word):
        '''Normalization used in pre-processing.

        - All words are lower cased
        - Digits in the range 1800-2100 are represented as !YEAR;
        - Other digits are represented as !DIGITS

        :rtype: str
        '''
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        '''
           Map tokens-in-contexts into a feature representation
        '''

        def add(name, *args):
            features.add('+'.join((name,) + tuple(args)))
 
        features = set()
        add('bias') # This acts sort of like a prior
        add('i suffix', word[-3:])
        add('i pref1', word[0])
        add('i-1 tag', prev)
        add('i-2 tag', prev2)
        add('i tag+i-2 tag', prev, prev2)
        add('i word', context[i])
        add('i-1 tag+i word', prev, context[i])
        add('i-1 word', context[i-1])
        add('i-1 suffix', context[i-1][-3:])
        add('i-2 word', context[i-2])
        add('i+1 word', context[i+1])
        add('i+1 suffix', context[i+1][-3:])
        add('i+2 word', context[i+2])
        return features




        

if __name__ == "__main__":
    ###################
    #corpus = brown.tagged_sents(categories='news')
    #test_instances = None
    ###################
    corpus = wsjtree2pos(WSJ_CORPUS_PATH)
    size = int(len(corpus)*0.9)
    train_instances = corpus[:size]
    dev_instances = corpus[size:]
    n_iter = 20
    model_loc = "models/pos-train-brown-news"
    model = SWTaggerModel()
    model.setup(train_instances)
    tagger = POSTagger(model)
    print 'BEGIN TRAINING'
    for i in range(n_iter):
        print 'Iteration: ' + str(i+1)
        print "shuffling training instances ..."
        random.shuffle(train_instances)
        tagger.train(train_instances)
        tagger.perceptron.average_weight()
        model.save_model(model_loc+'-iter'+str(i+1)+'.m')
        tagger.tag(dev_instances)
    print 'DONE TRAINING'
















