import sys
import os
from sklearn.datasets import load_svmlight_file,load_svmlight_files
from sklearn.cross_validation import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.linear_model import SGDClassifier, Perceptron,PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import ElasticNet,LogisticRegression
import numpy as np

#Algorithms
algoNames=["PA","KNN","LSVM","LogReg","RF","GBM"]
algoSetups=[PassiveAggressiveClassifier(random_state=0),KNeighborsClassifier(),LinearSVC(random_state=0),LogisticRegression(random_state=0),RandomForestClassifier(random_state=0),GradientBoostingClassifier(random_state=0,n_estimators=10)]


files = os.listdir("ReducedDatasets/")
#files = sys.argv[1].split(",")

variantes = ['chi2', 'chi2_et', 'chi2_fdr', 'chi2_fpr', 'chi2_fs', 'chi2_fwe', 'chi2_pca', 'et', 'et_chi2', 'et_fdr', 'et_fpr', 'et_fs', 'et_fwe', 'et_pca', 'fdr', 'fdr_chi2', 'fdr_et', 'fdr_fpr', 'fdr_fs', 'fdr_fwe', 'fdr_pca', 'fpr', 'fpr_chi2', 'fpr_et', 'fpr_fdr', 'fpr_fs', 'fpr_fwe', 'fpr_pca', 'fs', 'fs_chi2', 'fs_et', 'fs_fdr', 'fs_fpr', 'fs_fwe', 'fs_pca', 'full', 'fwe', 'fwe_chi2', 'fwe_et', 'fwe_fdr', 'fwe_fpr', 'fwe_fs', 'fwe_pca', 'hmbfs1.5', 'hmbfs1.5_chi2', 'hmbfs1.5_et', 'hmbfs1.5_fdr', 'hmbfs1.5_fpr', 'hmbfs1.5_fs', 'hmbfs1.5_fwe', 'hmbfs1.5_pca', 'hmbfs2.0', 'hmbfs2.0_chi2', 'hmbfs2.0_et', 'hmbfs2.0_fdr', 'hmbfs2.0_fpr', 'hmbfs2.0_fs', 'hmbfs2.0_fwe', 'hmbfs2.0_pca', 'pca']

for cadaF in files:
  #Save File
  salida=open("LogResults/"+cadaF+".csv",'w')
  salida.write("Dataset,FSmethod,NumFeats,CLFmethod,Accuracy,Precision,Recall\n")
  for cadaVariante in variantes:
    nomDS = cadaF+"_"+cadaVariante
    
    #Load the data
    print "\tDataset: " + nomDS
    X_sparse, y_full = load_svmlight_file("ReducedDatasets/"+nomDS)
    X_full = X_sparse.toarray()

    #Configuration
    numCases = len(y_full)
    dataLabels = list(set(y_full))
    numFeatFull = X_full.shape[1]
    print "\t Feats:" + str(numFeatFull)   
   
    for x in range(0,len(algoNames)):
      algoN=algoNames[x]
      algoS=algoSetups[x]
      
      loo = LeaveOneOut(numCases)
      
      y_p=[-1]*numCases
	  
      for train_index, test_index in loo:
	X_train, X_test = X_full[train_index], X_full[test_index]
	y_train, y_test = y_full[train_index], y_full[test_index]
	
	algoS.fit(X_train,y_train)
	y_p[test_index] = algoS.predict(X_test)
      
      y_pred = np.array(y_p) 
      aveAcc = accuracy_score(y_full, y_pred)
      avePre = precision_score(y_full, y_pred, labels=dataLabels, pos_label=None, average='macro')
      aveRec = recall_score(y_full, y_pred, labels=dataLabels, pos_label=None, average='macro')
      
      #print "\t Feats:" + str(numFeatFull)
      print "\t Algo:" +algoNames[x]
      print "\t  Acc:" + str(aveAcc)
      print "\t  Pre:" + str(avePre)
      print "\t  Rec:" + str(aveRec)
      
      salida.write(cadaF+","+cadaVariante+","+str(numFeatFull)+","+algoNames[x]+","+str(aveAcc)+","+str(avePre)+","+str(aveRec)+"\n")
  salida.close()

    
  
  
  
  
