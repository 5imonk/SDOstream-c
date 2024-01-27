from dSalmon.trees import MTree
import numpy as np

tree = MTree()

# insert a point [1,2,3,4] with key 5
tree[5] = [1,2,3,4]

# insert some random test data
X = np.random.rand(1000,4)
inserted_keys = tree.insert(X)

# delete every second point
del tree.ix[::2]

# Set the coordinates of the point with the lowest key
tree.ix[0] = [0,0,0,0]

# find the 3 nearest neighbors to [0.5, 0.5, 0.5, 0.5]
neighbor_keys, neighbor_distances, _ = tree.knn([.5,.5,.5,.5], k=3)
print ('Neighbor keys:', neighbor_keys)
print ('Neighbor distances:', neighbor_distances)

# find all neighbors to [0.5, 0.5, 0.5, 0.5] within a radius of 0.2
neighbor_keys, neighbor_distances, _ = tree.neighbors([.5,.5,.5,.5], radius=0.2)
print ('Neighbor keys:', neighbor_keys)
print ('Neighbor distances:', neighbor_distances)