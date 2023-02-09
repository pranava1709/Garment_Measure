import pandas as pd 
import scipy as sp
from scipy import spatial
from matplotlib import pyplot as plt
import numpy as np
import cv2
df = pd.read_csv("coord5.csv")

               
points = pd.DataFrame(df, columns=["x", "y"]).astype(float)
matrix = sp.spatial.distance_matrix(points, points)
print(matrix)


np.savetxt("matrix.txt",matrix)


print(points)

arr = np.asarray(points)


with open("pts1.txt",'a') as l:
    l.write(str(arr))
plt.draw()

plt.savefig("final_image1.png")
plt.show()
file  = pd.read_csv("matrix.txt")
with open("coord5.csv",'r') as f:
    for row in f:
        #print(type(row))
        #row1 = pd.DataFrame(row)
        #row1.to_csv("row"+str(t)+'.csv')
        print(row)
        with open("finalcoord.txt",'a') as g:
            g.write("[" + str(row) + "]"+',')
            g.write("\n")




