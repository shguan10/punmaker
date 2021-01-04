# this file will contain utility functions
import sys
sys.path.append("..")

import eng_to_ipa_to_eng as ipa
from tqdm import tqdm
import edlib
import pickle as pk
import random
import torch

import numpy as np
def editdistance(s,t):
  result = edlib.align(s,t,task="path")
  return result['editDistance']

def getdatapoint(wipa,edit_ratio = 0.5,numcands = 5):
  # generate the datapoint for this ipa word

  # randomly sample from the dictionary until you get what you need
  totalipa = ipa.totalipa()
  foundFar = 0
  cands = []
  closewipa = None
  while not ((closewipa is not None) and foundFar==numcands-1):
    ind = random.randint(0,totalipa-1)
    wipacand = ipa.wipaind(ind)

    ed = editdistance(wipacand,wipa)
    edit_threshold = edit_ratio * len(wipacand)
    if ed <= edit_threshold:
      if closewipa is not None:
        continue
      else:
        cands.append(wipacand)
        closewipa = wipacand
    else:
      if foundFar < numcands-1:
        foundFar+=1
        cands.append(wipacand)
      else: continue

  return (wipa,cands,closewipa)

def gendataset(numpoints=10000,edit_ratio=0.5,numcands=5):
  dataset = []
  totalipa = ipa.totalipa()
  
  for i in tqdm(range(numpoints)):
    ind = random.randint(0,totalipa-1)
    wipa = ipa.wipaind(ind)


    dataset.append(getdatapoint(wipa,edit_ratio=edit_ratio,numcands=numcands))
    # print(ipa.convert2eng(wipa),ipa.convert2eng(dataset[-1][-1]))

    # pdb.set_trace()

  numtrain = int(0.8*numpoints)
  numval = int(0.1*numpoints)

  trainset = dataset[:numtrain]
  valset = dataset[numtrain:numtrain+numval]
  testset = dataset[numtrain+numval:]

  with open("data/dataset_"+str(edit_ratio)+".pk","wb") as f:
    pk.dump((trainset,valset,testset),f)

import pdb

def zeropad(word,maxlen):
  word = list(word)
  wlen = len(word)
  result = [0 for _ in range(maxlen)]
  result[:wlen] = word
  # print(len(result))
  return result

def onehot(charnum,totalchars):
  return [i==charnum for i in range(totalchars)]

def prepdataset(edit_ratio=0.5):
  dname = "data/dataset_"+str(edit_ratio)
  with open(dname+".pk","rb") as f:
    data = pk.load(f)

  (trainset,valset,testset) = data
  # each of the train,val,test sets must be tuples of queries,cands,labels
  # queries has shape (wordlen,dataset_size,indim)
  # cands has shape (numcands,wordlen,dataset_size,indim)
  # labels has shape (dataset_size,)

  # find the length of the longest word in the dataset
  numtrain = len(trainset)
  numval = len(valset)
  numtest = len(testset)

  dataset = list(trainset)
  dataset.extend(valset)
  dataset.extend(testset)

  qs = [d[0] for d in dataset]
  cs = [d[1] for d in dataset]
  ys = np.array([np.array([c==d[2] for c in d[1]]).argmax() for d in dataset])

  lenqs = [len(q) for q in qs]
  lencs = [len(cand) for cl in cs for cand in cl]

  maxlen = max(max(lenqs),max(lencs))
  # pdb.set_trace()

  # translate each char into one-hot
  # first we must know how many chars there are
  chars = set()

  for q in qs:
    for char in q: chars.add(char)
  for clist in cs:
    for cand in clist:
      for char in cand:
        chars.add(char)

  # for each char in chars, associate a single number to it
  # the number 0 represents the STOP special char
  num2char = [0]+list(chars)
  char2num = {num2char[n]:n for n in range(len(num2char))}

  with open(dname+"_chardict.pk","wb") as f:
    pk.dump((num2char,char2num),f)

  # translate each char to its number
  qs = [[char2num[char] for char in q] for q in qs]
  cs = [[[char2num[char] for char in cand]for cand in cl]for cl in cs]

  # now zeropad
  qs = [zeropad(q,maxlen) for q in qs]
  cs = [[zeropad(c,maxlen)for c in cl]for cl in cs]

  # now one-hot encode each char
  qs = np.array([[onehot(cnum,len(num2char)) for cnum in q] for q in qs])
  cs = np.array([[[onehot(cnum,len(num2char)) for cnum in cand]for cand in cl]for cl in cs])

  # qs has shape (dsize,maxlen,numchars)
  # cs has shape (dsize,numcands,maxlen,numchars)

  qs = np.transpose(qs,axes=[1,0,2])
  cs = np.transpose(cs,axes=[1,2,0,3])

  qs = torch.tensor(qs).float()
  cs = torch.tensor(cs).float()
  ys = torch.tensor(ys)

  # now qs and cs are in the right shape
  trainset = (qs[:,:numtrain,:],cs[:,:,:numtrain,:],ys[:numtrain])
  valset = (qs[:,numtrain:numtrain+numval,:],cs[:,:,numtrain:numtrain+numval,:],ys[numtrain:numtrain+numval])
  testset = (qs[:,numtrain+numval:,:],cs[:,:,numtrain+numval:,:],ys[numtrain+numval:])

  with open(dname+"_formatted.pk","wb") as f:
    pk.dump((trainset,valset,testset),f)

def onehot2not(t):
  # assumes t has the onehot dim as the zeroth dim
  return torch.argmax(t,dim=0)

def getdatasetlengths(edit_ratio=0.4):
  dname = "data/dataset_"+str(edit_ratio)
  with open(dname+"_formatted.pk","rb") as f: data = pk.load(f)

  results = []

  for dset in data:
    subresult = []
    for ind,d in enumerate(dset):
      if ind==0:
        # ind ==0, queries
        maxlen,numsamples,numtokens = d.shape
        d = d.transpose(0,2)
        d = onehot2not(d)
        d = d.transpose(0,1)
        # now d is shape maxlen,numsamples
        zeros = (d==0).to(torch.float)
        ones = torch.ones((1,numsamples))
        catted = torch.cat((zeros,ones),dim=0) # now every sample has at least one 1

        lengths = np.argmax(catted.numpy(),axis=0) # for some reason torch argmax doesn't return the first maximal element, as advertised

        subresult.append(lengths)
      if ind==1:
        # ind ==1, cands
        numcands,maxlen,numsamples,numtokens = d.shape
        d = d.transpose(0,3)
        d = onehot2not(d)
        d = d.transpose(0,2)
        d = d.transpose(1,2)
        # now d is shape numcands,maxlen,numsamples
        zeros = d==0
        ones = torch.ones((numcands,1,numsamples))
        catted = torch.cat((zeros,ones),dim=1) # now every sample has at least one 1

        lengths = np.argmax(catted.numpy(),axis=1)

        subresult.append(lengths)
      if ind==2: continue
      # ind ==2, labels
    results.append(subresult)

  with open(dname+"_formatted_lengths.pk","wb") as f: pk.dump(results,f)

if __name__ == '__main__':
  # gendataset(edit_ratio=0.4)
  # prepdataset(edit_ratio=0.5)
  getdatasetlengths(edit_ratio=0.4)
  # print("hello")