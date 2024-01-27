from dSalmon import clustering, outlier
import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import minmax_scale

import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def sample_size (N, s, e):
	z=1.96
	num = N * pow(z,2) * pow(s,2)
	den = (N-1) * pow(e,2) + pow(z,2) * pow(s,2)
	n = int(math.floor(num/den))
	return n

# Let scikit-learn fetch and preprocess some stream data
kddcup = fetch_kddcup99()
X = minmax_scale(np.delete(kddcup.data, (1,2,3), 1))

size = 5000
# Randomly sample k entries from Xp
random_indices = np.random.choice(X.shape[0], size, replace=False)
X = X[random_indices]


X = np.random.rand(size, 3)

[m, n] = X.shape
print((m,n))
Xt = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
Xp = pca.fit_transform(Xt)
sigma = np.std(Xp)
if sigma<1:
    sigma=1
error = 0.1*np.std(Xp)
k = sample_size( m, sigma, error )


# Perform outlier detection using a Robust Random Cut Forest
T = 1000
print(k)
# k = 100
classifier = clustering.SDOcluststream(k=k, T=T)

# Initialize an array to store the labels
all_labels = []
block_size = size
# Iterate through the data in blocks
for i in range(0, X.shape[0], block_size):
    chunk = X[i:i + block_size, :]

    # Fit and predict using the current block (chunk)
    labels = classifier.fit_predict(chunk)

    # Append the labels to the list
    all_labels.append(labels)

# Concatenate the labels from all blocks
final_labels = np.concatenate(all_labels)


# labels = classifier.fit_predict(X)

obs = classifier.get_observers()
for o in obs:
      print(o)
# print(labels)
print(all_labels)

# detector = outlier.SDOstream(k=k, T=T);
# scores = detector.fit_predict(X)

# obs = detector.get_observers()
# for o in obs:
#       print(o)
