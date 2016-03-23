# Imports
from sklearn import preprocessing
import numpy as np

# Html 4.0 Web Colors
coloresWeb = [
 [0,0,0],
 [0,0,128],
 [0,128,0],
 [0,128,128],
 [128,0,0],
 [128,0,128],
 [128,128,0],
 [192,192,192],
 [128,128,128],
 [0,0,255],
 [0,255,0],
 [0,255,255],
 [255,0,0],
 [255,0,255],
 [255,255,0],
 [255,255,255]
]

support=None
colorsForImage=None

def getSupport():
  return support

def getImageData():
  return colorsForImage[:] #Copy?
  #return colorsForImage


# Color quantization
def getBaseColor(rValue=128, gValue=128, bValue=128):
  allDistances=[765]*16
  for x in range(0,16):
    valoresColor = coloresWeb[x]
    allDistances[x]= ((valoresColor[0]-rValue)**2 + (valoresColor[1]-gValue)**2 + (valoresColor[2]-bValue)**2)**0.5
    #allDistances[x]= (abs(valoresColor[0]-rValue) + abs(valoresColor[1]-gValue) + abs(valoresColor[2]-bValue))
  return allDistances.index(min(allDistances))

def hmbFitTransform(oriX,y,thVal):
  global support
  global colorsForImage
  colorsForImage=None
  support=None
  
  X_original = np.copy(oriX)
  X = np.copy(oriX)

  numClases = len(set(y))
  numFeats = len(X[0])
  numInstances = len(y)

  #Scale to 0-255
  #print "Scale dataset..."
  min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255), copy=False)
  X = min_max_scaler.fit_transform(X,y)

  #Create new dataset compated
  #print "Running HmbFS..."

  #Create empty dataset
  numGrupos = numFeats/3
  if numFeats%3 > 0:
    numGrupos+=1
    
  coloredDataset = np.zeros((numInstances, numGrupos), dtype=np.int8) #(instances,features)

  #Build colored dataset
  #print "This is slow..."

  for ins in range(0,numInstances):
    for gro in range(0,numGrupos):
      if (gro*3)+2 < numFeats:
        coloredDataset[ins][gro]= getBaseColor(X[ins][gro*3],X[ins][(gro*3)+1],X[ins][(gro*3)+2])
      else:
        coloredDataset[ins][gro]= getBaseColor(X[ins][-3],X[ins][-2],X[ins][-1])

  #print "Finally..."
  
  usefulFeats = np.zeros(numFeats,dtype=bool)
  th=thVal

  for cF in range(0,numGrupos):
    colorDistrib = np.zeros((numClases, 16), dtype=np.float16) #Because 16 base colors
    for cI in range(0,numInstances):
      colorDistrib[y[cI],coloredDataset[cI][cF]]+=1.0
    
    #print colorDistrib[0]
    #print colorDistrib[1]
    #Ok, now I know the distribution
    for cA in range(0,numClases):
      for cB in range(0,numClases):
	if cA != cB:
	  if ( (max(colorDistrib[cA])/sum(colorDistrib[cA])) >=  (th * (colorDistrib[cB][colorDistrib[cA].argmax()]/sum(colorDistrib[cB]))) ):
	    if ((cF*3)+2) < numFeats:
	      usefulFeats[(cF*3)]=True
	      usefulFeats[(cF*3)+1]=True
	      usefulFeats[(cF*3)+2]=True
	    else:
	      usefulFeats[-1]=True
	      usefulFeats[-2]=True
	      usefulFeats[-3]=True
	      
  X_reduced = X_original[:, usefulFeats]
  support=usefulFeats
  colorsForImage=coloredDataset
  return X_reduced



