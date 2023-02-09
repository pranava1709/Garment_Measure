import numpy as np
param = np.load("K.npy")
param1 = np.load("dist.npy")
param2 = np.load("ret.npy")
param3 = np.load("rvecs.npy")
param4 = np.load("tvecs.npy")
matrix  = param * param2
caminv  = np.linalg.inv(matrix)
#print(param)
#print(param1)
#print(param2)
#print(param3)
print(caminv)
np.savetxt("scale.txt",caminv)
#with open("scaling_matrix.txt",'a') as f:
	#f.write()
