import numpy as np

old = np.load("embeddings/facenet/dustin_facenet_20261026.npy")
new = np.load("embeddings/facenet/dustin_facenet.npy")
print(old.shape, new.shape)
print(np.allclose(old, new))
