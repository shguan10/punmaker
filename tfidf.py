"""
IMPORTANT
you must call getcodes() once in order to use this module like a library
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np

import pickle as pk

import sys
sys.path.append("..")

import eng_to_ipa_to_eng as ipa

codes = {"allcodes":None,"vocab":None,"idfs":None}

def getcodes():
  with open("codes/ngram3.pk","rb") as f:
    allcodes,vocab,idfs = pk.load(f)  
  codes['allcodes'] = allcodes
  codes['vocab'] = vocab
  codes['idfs'] = idfs
  return codes

def gencodes():
  ipalist = ipa.ipa

  vectorizer = TfidfVectorizer(analyzer=ngrams)
  allcodes = vectorizer.fit_transform(ipalist)
  vocab = vectorizer.vocabulary_
  idfs = vectorizer.idf_
  # print(vectorizer.get_feature_names())
  with open("codes/ngram3.pk","wb") as f:
    pk.dump((allcodes,vocab,idfs),f)

def ngrams(s, length=3):
    s = s.lower()
    
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    s = re.sub(rx, '', s)
    s = re.sub(' +',' ',s).strip() # get rid of multiple spaces and replace with a single
    
    pad = (length-1)*' '
    s = pad+ s +pad # pad names for ngrams...
    
    result = [s[i:i+length] for i in range(len(s)-length+1)]
    
    return result

def count(gram,list_of_grams):
    return sum([g==gram for g in list_of_grams])

def norm(vec):
    return ((vec**2).sum())**0.5
def weng2vec(weng):
    wipa = ipa.convert2ipa(weng)
    return wipa2vec(wipa)

def word_lookup_vec(weng):
    wipa = ipa.convert2ipa(weng)
    ind = np.argmax([p==wipa for p in ipa.ipa])
    assert ipa.wipaind(ind)==wipa
    return allcodes[ind]

def weng2cands(weng,numcands=5):
  wipa = ipa.convert2ipa(weng)
  return wipa2cands(wipa,numcands=numcands)

def wengs2cands(wengs,numcands=5):
  wipas = [ipa.convert2ipa(weng) for weng in wengs]
  return wipas2cands(wipas,numcands=numcands)

def wipa2vec(wipa):
  wipa_gs = ngrams(wipa)
  counts = [0 for _ in codes['vocab']]
  
  for gram in codes['vocab']:
      c = count(gram,wipa_gs)
      ind = codes['vocab'][gram]
      counts[ind] = c

  counts = np.array(counts)
  
  tfs = counts / len(wipa_gs)
  tfidfs = tfs * codes['idfs']
  
  norm = ((tfidfs**2).sum())**0.5
  
  wordvec = tfidfs / norm
  
  return wordvec.reshape(-1,1)

def wipa2cands(wipa,numcands=5):
  wordvec = wipa2vec(wipa)
  sims = codes['allcodes'] @ wordvec
  
  sims = sims.flatten()
  
  inds = np.argpartition(sims, -numcands)[-numcands:]
  
  wengs = [ipa.convert2eng(ipa.ipa[ind]) for ind in inds]
  return wengs
import pdb
def wipas2cands(wipas,numcands=5):
  wordvecs = np.array([wipa2vec(wipa).flatten() for wipa in wipas])
  sims = codes['allcodes'] @ (wordvecs.transpose())
  # pdb.set_trace()
  
  words = []

  for ind in range(len(wipas)):
    choices = sims[:,ind]
    choices = choices.flatten()

    inds = np.argpartition(choices, -numcands)[-numcands:]
  
    wengs = [ipa.convert2eng(ipa.ipa[ind]) for ind in inds]
    
    words.append(wengs)
  return words

if __name__ == '__main__':
  # print(wengs2cands(["carrot","happy","birthday"]))
  gencodes()