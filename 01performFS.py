from sklearn.datasets import load_svmlight_file,load_svmlight_files
from sklearn.decomposition import PCA,KernelPCA
from sklearn.feature_selection import SelectPercentile, chi2, SelectFpr, f_classif, RFECV, RFE, SelectFdr, SelectFwe, SelectKBest
from sklearn.datasets import dump_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import ExtraTreesClassifier
from hmbfs import *

def printStats(nom,X,y):
  print " ==========="
  print " "+nom
  print " Instances:" + str(len(y))
  print " Features:" + str(len(X[0]))
  print " Val I1F1:" + str(X[0][0])
  
datasetNames=["alon","golubScaled","suScaled","shippScaled","tian","chowdary","singhScaled"]

for eachDataset in datasetNames:
  print eachDataset
  X_sparse, y_full = load_svmlight_file("OriginalDatasets/"+eachDataset)
  X_full = X_sparse.toarray()
  methodsNames=["full","pca","chi2","fs","et","fpr","fdr","fwe"]
  methodsFS=[SelectPercentile(chi2, percentile=100),PCA(n_components=0.9),SelectPercentile(chi2, percentile=75),SelectPercentile(f_classif, percentile=75),ExtraTreesClassifier(random_state=0),SelectFpr(),SelectFdr(),SelectFwe()]
  for x in range(0,len(methodsFS)):
    fsm=methodsFS[x]
    fsm.fit(X_full,y_full)
    X_redu=fsm.transform(X_full)
    
    #Some algorithms fail and select 0 features, lets fix that
    if len(X_redu[0])<3:
      tmpMethod=SelectKBest(chi2,k=3)
      tmpMethod.fit(X_full,y_full)
      X_redu=tmpMethod.transform(X_full)
    
    fileOut=open("ReducedDatasets/"+eachDataset+"_"+methodsNames[x],'wb')
    dump_svmlight_file(X_redu,y_full,fileOut,zero_based=False)
    fileOut.close()
    
    goodFeats=[]
    if methodsNames[x]=="et":
      numF=1
      for cadaFeat in fsm.feature_importances_:
	if abs(cadaFeat) >0:
	  goodFeats.append(numF)
	numF+=1
    elif methodsNames[x]=="pca":
      for nf in range(0,fsm.n_components_):
	goodFeats.append(nf)
    else:
      goodFeats = fsm.get_support(indices=True)
      
    fileOutFeats=open("SelectedFeatures/"+eachDataset+"_"+methodsNames[x],'w')
    for cadaF in goodFeats:
      fileOutFeats.write(str(cadaF+1)+"\n")
    fileOutFeats.close()
   
  #HmbFS code
  for th in [1.5,2.0]:
    X_redu=hmbFitTransform(X_full,y_full,th)
    
    fileOut=open("ReducedDatasets/"+eachDataset+"_hmbfs"+str(th),'wb')
    dump_svmlight_file(X_redu,y_full,fileOut,zero_based=False)
    fileOut.close()
    fileOutFeats=open("SelectedFeatures/"+eachDataset+"_hmbfs"+str(th),'w')
    goodFeats=getSupport()
    numF=1
    for cadaF in goodFeats:
      if cadaF==True:
	fileOutFeats.write(str(numF)+"\n")
      numF+=1
    
  print "Done!"
    
#Ensembles
sourceDS = ["hmbfs1.5","hmbfs2.0","chi2","fs","et","fpr","fdr","fwe"]
ensembles = ["pca","chi2","fs","et","fpr","fdr","fwe"]
ensembleFS = [PCA(n_components=0.9),SelectPercentile(chi2, percentile=75),SelectPercentile(f_classif, percentile=75),ExtraTreesClassifier(random_state=0),SelectFpr(),SelectFdr(),SelectFwe()]
for eachDataset in datasetNames:
  for eachEnsemble  in sourceDS:
    ensembleNameDS=eachDataset+"_"+eachEnsemble
    print ensembleNameDS
    X_sparse, y_full = load_svmlight_file("ReducedDatasets/"+ensembleNameDS)
    X_full = X_sparse.toarray()

    for x in range(0,len(ensembles)):
      if eachEnsemble!=ensembles[x]: #So, we do not do, Chi2->Chi2 FS
	fsm=ensembleFS[x]
	fsm.fit(X_full,y_full)
	X_redu=fsm.transform(X_full)
	
	#Some algorithms fail and select 0 features, lets fix that
	if len(X_redu[0])<3:
	  tmpMethod=SelectKBest(chi2,k=3)
	  tmpMethod.fit(X_full,y_full)
	  X_redu=tmpMethod.transform(X_full)
	
	printStats(ensembles[x],X_redu,y_full)
	fileOut=open("ReducedDatasets/"+ensembleNameDS+"_"+ensembles[x],'wb')
	dump_svmlight_file(X_redu,y_full,fileOut,zero_based=False)
	fileOut.close()
	
	goodFeats=[]
	if ensembles[x]=="et":
	  numF=1
	  for cadaFeat in fsm.feature_importances_:
	    if abs(cadaFeat) >0:
	      goodFeats.append(numF)
	    numF+=1
	elif ensembles[x]=="pca":
	  for nf in range(0,fsm.n_components_):
	    goodFeats.append(nf)
	else:
	  goodFeats = fsm.get_support(indices=True)
	  
	fileOutFeats=open("SelectedFeatures/"+ensembleNameDS+"_"+ensembles[x],'w')
	for cadaF in goodFeats:
	  fileOutFeats.write(str(cadaF+1)+"\n")
	fileOutFeats.close()

  
  
