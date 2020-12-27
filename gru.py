import torch
from torch import nn

import pdb
import pickle as pk
import numpy as np

import time

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

  def encode(self,wipa):
    # wipa has been translated but not one-hot encoded
    totalchars = len(self.char2num)

    wipatranslated = [onehot(charnum,totalchars) for charnum in wipa]

    wipatranslated = torch.tensor(wipatranslated).float()

    # wipatranslated has shape (wordlen,totalchars)

    wipatranslated = wipatranslated.reshape(-1,1,totalchars)

    wresult = None

    with torch.no_grad():
      wipaf = self.f(wipatranslated)[1].cpu().numpy()
      wipaf = wipaf.reshape((self.numlayers,1,-1,self.hdim))
      wipaf = wipaf[-1,0,:,:].flatten()
      
      wipag = self.g(wipatranslated)[1].cpu().numpy()
      wipag = wipag.reshape((self.numlayers,1,-1,self.hdim))
      wipag = wipag[-1,0,:,:].flatten()
      
    wresult = np.concatenate((wipaf,wipag))
    return wresult

  def bulkencode(self,wipas):
    # assumes wipas has shape (dataset_size,wordlen)
    # ie wipas is zeropadded but not one-hot encoded
    totalchars = len(self.char2num)

    wipastranslated = [[onehot(charnum,totalchars) for charnum in wipa] for wipa in wipas]

    wipastranslated = torch.tensor(wipastranslated).float()

    # wipastranslated has shape (dataset_size,wordlen,totalchars)
    wipastranslated = wipastranslated.transpose(0,1)

    wresult = None

    with torch.no_grad():
      wipaf = self.f(wipastranslated)[1].cpu().numpy()
      wipaf = wipaf.reshape((self.numlayers,1,-1,self.hdim))
      wipaf = wipaf[-1,0,:,:]

      wipag = self.g(wipastranslated)[1].cpu().numpy()
      wipag = wipag.reshape((self.numlayers,1,-1,self.hdim))
      wipag = wipag[-1,0,:,:]

    print(wipaf.shape,wipag.shape)

    wresult = np.concatenate((wipaf,wipag),axis=1)
    print(wresult.shape)

    # make sure to normalize each encoding by its l2 norm
    l2norms = (((wresult**2).sum(axis=1))**0.5).reshape(-1,1)

    pdb.set_trace()

    return wresult / l2norms

def check_num_correct(py,by):
  # by is shape (bsize,)
  # py is shape (bsize,numcands)
  (bsize,numcands) = py.shape
  return sum(py.argmax(1)==by)

def train_loop(model,optimizer,dataset,maxpatience = 20,bsize=32,verbose=False,numepochs=200,datastore=None):
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

  starttime = time.time()
  for epoch in range(numepochs):
    epochloss = 0
    trainacc = 0
    idxs = (np.random.rand(numbatches,bsize)*numtrain).astype(np.int64)

    for batch in idxs:
      optimizer.zero_grad()
      bqueries = train[0][:,batch,:]
      bcands = train[1][:,:,batch,:]
      blabels = train[2][batch]

      bqueries = bqueries.cuda()
      bcands = bcands.cuda()
      blabels = blabels.cuda()

      py = model.forward(bqueries,bcands)
      loss = criterion.forward(py,blabels)
      epochloss+=loss

      loss.backward()
      optimizer.step()

      trainacc += check_num_correct(py.detach().cpu().numpy(),blabels.cpu().numpy())

    trainacc /= numtrain
    avgsampleloss = epochloss/numtrain

    with torch.no_grad():
      vqueries = val[0].cuda()
      vcands = val[1].cuda()
      vlabels = val[2].cuda()
      numval = len(vlabels)

      valps = model.forward(vqueries,vcands)
      valloss = criterion(valps,vlabels)/numval
      # pdb.set_trace()
      valacc = check_num_correct(valps.cpu(),vlabels.cpu()).item()/numval
    if datastore is not None: 
      with torch.no_grad():
        numtest = len(test[2])

        tqueries = test[0].cuda()
        tcands = test[1].cuda()
        tlabels = test[2].cuda()

        testps = model.forward(tqueries,tcands)
        testloss = criterion(testps,tlabels)/numtest
        testacc = check_num_correct(testps.cpu(),test[2].cpu()).item()/numtest

      datastore.append(((avgsampleloss,valloss,testloss),(trainacc,valacc,testacc)))

    if verbose:
      elapsed_time = time.time() - starttime
      print("epoch: ",epoch,"/",numepochs-1,
        ", trainloss: %.4f" %avgsampleloss, 
        ", trainacc: %.4f" %trainacc, 
        "valloss: %.4f" %valloss,
        "valacc: %.4f" %valacc,
        "elapsed_time: %d"%int(elapsed_time),
        end="\r")
    
    if prevmin is None or valloss < prevmin: 
      patience = maxpatience
      prevmin = valloss
    else: patience -= 1
    if patience <=0: break
  print("\ntestloss: %.4f" %testloss,
        "testacc: %.4f" %testacc)
  # if verbose: print("\n")

  return valacc

def train_driver(edit_ratio=0.4): # CHOO CHOO
  dname = "data/dataset_"+str(edit_ratio)
  with open(dname+"_formatted.pk","rb") as f:
    dataset = pk.load(f)

  model = Deep_Classifier(hdim=10,edit_ratio=edit_ratio).cuda()

  optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

  datastore = []

  valacc = train_loop(model,
                        optimizer,
                        dataset,
                        maxpatience = 200,
                        bsize=32,
                        verbose=True,
                        numepochs=200,
                        datastore=datastore)


  torch.save(model.state_dict(),"models/model_"+str(edit_ratio)+"_valacc_"+str(valacc)+".pt")

if __name__ == '__main__':
  train_driver(edit_ratio=0.4)