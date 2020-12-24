# this file will contain utility functions
import sys
sys.path.append("..")

import eng_to_ipa_to_eng as ipa
from tqdm import tqdm
import edlib
import pickle as pk
import random

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
import pdb
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

if __name__ == '__main__':
  gendataset(edit_ratio=0.4)