import numpy as np
from sklearn.cluster import KMeans

def cluster_motifs(cams: np.ndarray, k: int = 8):
    X = cams.squeeze(1).reshape(cams.shape[0], -1)
    km = KMeans(n_clusters=k, n_init=10, random_state=1337)
    labels = km.fit_predict(X)
    return labels
