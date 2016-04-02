from sklearn.datasets import load_svmlight_file,load_svmlight_files
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA


#X, y = load_svmlight_file("OriginalDatasets/"+"chowdary")
X, y = load_svmlight_file("ReducedDatasets/"+"chowdary_hmbfs2.0")
#X, y = load_svmlight_file("ReducedDatasets/"+"chowdary_et")
#X, y = load_svmlight_file("ReducedDatasets/"+"chowdary_fwe")
#X, y = load_svmlight_file("ReducedDatasets/"+"chowdary_hmbfs1.5_fwe")

X=X.toarray()
target_names = ["Cancer","Normal"]

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

plt.figure()

for c, i, target_name, marcador in zip("rg", [0, 1], target_names,["x","o"]):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, s=400, marker=marcador, label=target_name)


plt.legend()

#plt.title('Chowdary PCA: Full Features')
plt.title('Chowdary PCA: HmbFS Features')
#plt.title('Chowdary PCA: Et Features')
#plt.title('Chowdary PCA: Fwe Features')
#plt.title('Chowdary PCA: HmbFS+Fwe Features')

plt.show()