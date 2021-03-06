import sys
sys.path.append("..")

import eng_to_ipa_to_eng as ipa
import gru
import pickle as pk

import util
import torch
from scipy.stats import expon
import scipy as sp
from scipy import stats
import numpy as np
import pdb

import tfidf

def prepare_encodings(edit_ratio=0.4):
  dname = "data/dataset_"+str(edit_ratio)
  mname = "models/model_"+str(edit_ratio)+"_valacc_0.996.pt"# TODO the valacc should be automatically the max in the dir
  cname = "codes/codes_"+str(edit_ratio)+"_valacc_0.996.pk"# TODO this should automatically correspond to the model chosen

  with open(dname+"_chardict.pk","rb") as f:
    (num2char,char2num) = pk.load(f)
  totalchars = len(num2char)
  ipalist = ipa.ipa

  lengths = np.array([len(wipa) for wipa in ipalist])

  ipalist = [[char2num[char] for char in wipa] for wipa in ipalist]

  maxlen = max([len(wipa) for wipa in ipalist])
  zeropadded = [util.zeropad(wipa,maxlen) for wipa in ipalist]
  zeropadded = [[util.onehot(charnum,totalchars) for charnum in wipa] 
                for wipa in zeropadded]
  zeropadded = np.array(zeropadded)
  # zeropadded has shape numwords,maxlen,totalchars
  zeropadded = zeropadded.swapaxes(0,1)
  # zeropadded has shape maxlen,numwords,totalchars

  lengths_sortindx = np.argsort(lengths)
  lengths_sortindx = np.array(lengths_sortindx[::-1])
  zeropadded = torch.tensor(zeropadded[:,lengths_sortindx,:]).float()
  lengths = lengths[lengths_sortindx]

  # pdb.set_trace()

  zeropadded = torch.nn.utils.rnn.pack_padded_sequence(zeropadded,lengths)

  model = gru.Deep_Classifier(hdim=10,numlayers=1,edit_ratio=edit_ratio)# TODO get hdim and numlayers from somewhere
  model.load_state_dict(torch.load(mname))
  model.eval()

  encoded = model.encode(zeropadded)

  inverse_sortindx = np.empty(lengths_sortindx.shape).astype(lengths_sortindx.dtype)
  for ind,val in enumerate(lengths_sortindx):
    inverse_sortindx[val] = ind

  # pdb.set_trace()

  encoded = encoded[inverse_sortindx,:]

  with open(cname,"wb") as f:
    pk.dump(encoded,f)

def close_word(weng,edit_ratio=0.4):
  wipa = ipa.convert2ipa(weng) # TODO should do something when weng is not in dict

  dname = "data/dataset_"+str(edit_ratio)
  mname = "models/model_"+str(edit_ratio)+"_valacc_0.996.pt"# TODO the valacc should be automatically the max in the dir
  cname = "codes/codes_"+str(edit_ratio)+"_valacc_0.996.pk"# TODO this should automatically correspond to the model chosen

  with open(dname+"_chardict.pk","rb") as f:
    (num2char,char2num) = pk.load(f)
  totalchars = len(num2char)

  wipanums = [char2num[char] for char in wipa]
  onehot = [util.onehot(charnum,totalchars) for charnum in wipanums]
  onehot = torch.tensor(onehot).float()
  onehot = onehot[:,None,:]
  # onehot has shape (wordlen,1,totalchars)

  model = gru.Deep_Classifier(hdim=10,numlayers=1,edit_ratio=edit_ratio)# TODO get hdim and numlayers from somewhere
  model.load_state_dict(torch.load(mname))
  model.eval()

  wenc = model.encode(onehot)

  with open(cname,"rb") as f:
    ipacodes = pk.load(f)

  # look through the entire ipacodes for the greatest cosine similarity to wenc
  widx = np.argmax(ipacodes@wenc.transpose())
  return ipa.convert2eng(ipa.wipaind(widx))

def random_parts(strlen):
  if strlen<=0: return None
  # exponential random variable
  dist = 0
  result = []
  while dist<strlen:
    var = expon.rvs(scale=3,size=1)
    var = int(var)
    if var == 0: continue
    dist += var
    if dist > strlen: dist = strlen
    result.append(dist)
  return result

def equal_parts(strlen,numparts=3):
  # numparts = int(sp.stats.norm(original_num_tokens-1,2).rvs())
  # numparts = 3

  len_each = int(strlen / numparts)
  if strlen <= 10: return [strlen]

  # pdb.set_trace()

  return [int(len_each*(i+1)) for i in range(numparts)] + ([strlen] if (strlen % numparts) else [])

def close_phrase(weng_phrase,edit_ratio=0.4):
  dname = "data/dataset_"+str(edit_ratio)
  mname = "models/model_"+str(edit_ratio)+"_valacc_0.996.pt"# TODO the valacc should be automatically the max in the dir
  cname = "codes/codes_"+str(edit_ratio)+"_valacc_0.996.pk"# TODO this should automatically correspond to the model chosen

  with open(dname+"_chardict.pk","rb") as f:
    (num2char,char2num) = pk.load(f)

  wengs = weng_phrase.split(" ")

  wipas = [ipa.convert2ipa(weng)for weng in wengs] # TODO should do something when weng is not in dict
  wipas = [char for wipa in wipas for char in wipa]

  part_ends = random_parts(len(wipas))
  part_begins = [0]+part_ends[:-1]

  # print(part_ends)

  qphrase = [wipas[b:e] for b,e in zip(part_begins,part_ends)]
  qphrase = [[char2num[char] for char in wipa] for wipa in qphrase]

  model = gru.Deep_Classifier(hdim=10,numlayers=1,edit_ratio=edit_ratio)# TODO get hdim and numlayers from somewhere
  model.load_state_dict(torch.load(mname))
  model.eval()

  qencs = []
  for wipanums in qphrase:
    wenc= model.encode(wipanums)
    qencs.append(wenc)
  qencs = np.array(qencs)

  with open(cname,"rb") as f:
    ipacodes = pk.load(f)

  csim = ipacodes@qencs.transpose()
  # pdb.set_trace()
  desired = csim.argmax(axis=0)

  wengs = []
  for windx in desired:
    wengs.append(ipa.convert2eng(ipa.wipaind(windx)))

  return wengs

def close_phrase_tfidf(weng_phrase,ngram_length = 3,numparts=3):
  # currently only supports ngram_length 3
  wengs = weng_phrase.split(" ")

  wipas = [ipa.convert2ipa(weng)for weng in wengs] # TODO should do something when weng is not in dict
  # pdb.set_trace()
  wipas = ''.join([char for wipa in wipas for char in wipa])

  # part_ends = equal_parts(len(wipas),numparts=numparts)
  part_ends = random_parts(len(wipas))
  part_begins = [0]+part_ends[:-1]

  # print(part_ends)

  # pdb.set_trace()

  wipas = [wipas[b:e] for b,e in zip(part_begins,part_ends)]


  return tfidf.wipas2cands(wipas,numcands=1)

if __name__ == '__main__':
  # prepare_encodings(edit_ratio=0.4)
  print(close_phrase("happy birthday to you"))
  # print(close_word("happy"))
  # tfidf.getcodes()
  # print(close_phrase_tfidf(input(),numparts=9))