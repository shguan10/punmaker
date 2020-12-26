import torch
from torch import nn

import pdb
import pickle as pk
import numpy as np
def onehot(charnum,totalchars):
  return [i==charnum for i in range(totalchars)]

class Deep_Classifier(torch.nn.Module):
  def __init__(self,hdim=50,numlayers=1,edit_ratio=0.4):
    torch.nn.Module.__init__(self)

    self.char2num = None
    self.num2char = None

    self.initchardict(edit_ratio=edit_ratio)

    indim = len(self.char2num)

    self.f = nn.GRU(indim,hdim,numlayers)
    self.g = nn.GRU(indim,hdim,numlayers)
    self.numlayers = numlayers
    self.hdim = hdim


  def forward(self,query,cands):
    # query has shape (wordlen,bsize,indim)
    # cands has shape (numcands,wordlen,bsize,indim)
    # this function returns shape (bsize,numcands)
    wordlen,bsize,indim = query.shape

    _,qenc = self.f(query)

    qenc = qenc.view(self.numlayers,1,bsize,self.hdim)
    qenc = qenc[-1,0,:,:]

    # qenc has shape (bsize,hdim)

    numcands,wordlen,bsize,maxchars = cands.shape
    cands = cands.transpose(0,1)
    cands = cands.reshape(wordlen,numcands*bsize,maxchars)

    _,candsenc = self.g(cands)

    # candsenc has shape (numlayers,1*numcands*bsize,hdim)
    candsenc = candsenc.view(self.numlayers,1,numcands,bsize,self.hdim)
    candsenc = candsenc.transpose(0,2)
    # candsenc has shape (numcands,numlayers,1,bsize,hdim)

    candsenc = candsenc[:,-1,0,:,:]

    # candsenc has shape (numcands,bsize,hdim)

    prod = qenc * candsenc

    # prod should have shape (numcands,bsize,hdim)

    prod = prod.sum(2)

    # now prod should have shape (numcands,bsize)

    return prod.transpose(0,1)

  def initchardict(self,edit_ratio=0.4):
    dname = "data/dataset_"+str(edit_ratio)
    with open(dname+"_chardict.pk","rb") as f:
      num2char,char2num = pk.load(f)
    self.num2char = num2char
    self.char2num = char2num

  def encode(self,wipa,edit_ratio=0.4):
    totalchars = len(self.char2num)

    wipatranslated = [self.char2num(char) for char in wipa]
    wipatranslated = [onehot(charnum,totalchars) for charnum in wipatranslated]

    wipatranslated = torch.tensor(wipatranslated)

    # wipatranslated has shape (wordlen,totalchars)

    wipatranslated = wipatranslated.reshpae(-1,1,totalchars)

    wresult = None

    with torch.no_grad():
      wipaf = self.f(wipatranslated).cpu().numpy().flatten()
      wipag = self.g(wipatranslated).cpu().numpy().flatten()

    wresult = np.concatenate(wipaf,wipag)
    return wresult


def check_num_correct(py,by):
  # by is shape (bsize,)
  # py is shape (bsize,numcands)
  (bsize,numcands) = py.shape
  return sum(py.argmax(1)==by)

def train_network(model,optimizer,dataset,maxpatience = 20,bsize=32,verbose=False,early_stop=0.001,numepochs=200,datastore=None):
  # each of the train,val,test sets must be tuples of queries,cands,labels
  # queries has shape (wordlen,dataset_size,indim)
  # cands has shape (numcands,wordlen,dataset_size,indim)
  # labels has shape (dataset_size,)
  train,val,test = dataset

  numtrain = len(train[2])
  numbatches = int(numtrain/bsize)

  patience = maxpatience
  prevmin = None
  criterion = torch.nn.CrossEntropyLoss()
  for epoch in range(numepochs):
    epochloss = 0
    trainacc = 0
    idxs = (np.random.rand(numbatches,bsize)*numtrain).astype(np.int64)

    for batch in idxs:
      optimizer.zero_grad()
      bqueries = train[0][:,batch,:]
      bcands = train[1][:,:,batch,:]
      blabels = train[2][batch]

      py = model.forward(bqueries,bcands)
      loss = criterion.forward(py,blabels)
      epochloss+=loss

      loss.backward()
      optimizer.step()

      trainacc += check_num_correct(py.detach().cpu().numpy(),blabels.cpu().numpy())

    trainacc /= numtrain
    avgsampleloss = epochloss/numtrain

    with torch.no_grad():
      vqueries = val[0]
      vcands = val[1]
      vlabels = val[2]
      numval = len(vlabels)

      valps = model.forward(vqueries,vcands)
      valloss = criterion(valps,vlabels)/numval
      # pdb.set_trace()
      valacc = check_num_correct(valps.cpu(),vlabels.cpu()).item()/numval
    if datastore is not None: 
      with torch.no_grad():
        numtest = len(test[2])
        testps = model.forward(test[0],test[1])
        testloss = criterion(testps,test[2])/numtest
        testacc = check_num_correct(testps.cpu(),test[2].cpu()).item()/numtest

      datastore.append(((avgsampleloss,valloss,testloss),(trainacc,valacc,testacc)))

    if verbose:
      print("epoch: ",epoch,"/",numepochs-1,
        ", trainloss: %.4f" %avgsampleloss, 
        ", trainacc: %.4f" %trainacc, 
        "valloss: %.4f" %valloss,
        "valacc: %.4f" %valacc,
        end="\r")
    
    if prevmin is None or valloss < prevmin: 
      patience = maxpatience
      prevmin = valloss
    else: patience -= 1
    if patience <=0: break
  print("\ntestloss: %.4f" %testloss,
        "testacc: %.4f" %testacc)
  # if verbose: print("\n")

def main(edit_ratio=0.4):
  dname = "data/dataset_"+str(edit_ratio)
  with open(dname+"_formatted.pk","rb") as f:
    dataset = pk.load(f)

  model = Deep_Classifier(hdim=10,edit_ratio=edit_ratio)

  optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

  datastore = []

  train_network(model,
                optimizer,
                dataset,
                maxpatience = 20,
                bsize=32,
                verbose=True,
                early_stop=0.001,
                datastore=datastore)


  torch.save(model.state_dict(),"models/model_"+str(edit_ratio)+".pt")

if __name__ == '__main__':
  main()