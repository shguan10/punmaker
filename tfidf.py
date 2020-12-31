from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np

import pickle as pk

import sys
sys.path.append("..")

import eng_to_ipa_to_eng as ipa

with open("codes/ngram3.pk","rb") as f:
  allcodes,vocab,idfs = pk.load(f)

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
    wipa_gs = ngrams(wipa)
    
    counts = [0 for _ in vocab]
    
    for gram in vocab:
        c = count(gram,wipa_gs)
        ind = vocab[gram]
        counts[ind] = c

    counts = np.array(counts)
    
    tfs = counts / len(wipa_gs)
    tfidfs = tfs * idfs
    
    norm = ((tfidfs**2).sum())**0.5
    
    wordvec = tfidfs / norm
    
    return wordvec.reshape(-1,1)

def word_lookup_vec(weng):
    wipa = ipa.convert2ipa(weng)
    ind = np.argmax([p==wipa for p in ipa.ipa])
    assert ipa.wipaind(ind)==wipa
    return allcodes[ind]

def word2cands(weng,numcands=5):
    wordvec = weng2vec(weng)
    
    sims = allcodes @ wordvec
    
    sims = sims.flatten()
    
    inds = np.argpartition(sims, -numcands)[-numcands:]
    
    wengs = [ipa.convert2eng(ipa.ipa[ind]) for ind in inds]
    return wengs

if __name__ == '__main__':
  print(word2cands("carrot"))