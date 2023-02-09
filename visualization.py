import cv2
from matplotlib import pyplot as plt
import numpy as np
image = cv2.imread("/home/cslabdp/Downloads/deepfashion2-kps-agg-finetune-master/HRNet-Human-Pose-Estimation/128B.jpg")
print(image.shape)
image = cv2.resize(image,(60,60))
                        
pts = np.array([[30.25,4.75
],
[30.25,8.25
],
[33.75,9.75
],
[36.75,8.75
],
[37.75,5.25
],
[21.75,12.25
],
[15.25,22.75
],
[2.25,44.75
],
[7.25,48.75
],
[16.25,39.75
],
[20.75,22.75
],
[19.75,27.25
],
[18.25,40.25
],
[18.25,51.75
],
[34.75,59.75
],
[48.75,53.25
],
[45.25,37.75
],
[45.25,30.25
],
[45.25,30.75
],
[48.25,37.75
],
[55.25,48.25
],
[61.25,45.25
],
[50.25,21.75
],
[44.25,12.25
],
[34.25,6.75
],
[30.25,4.75
],
[30.25,8.25
],
[33.75,9.75
],
[36.25,8.75
],
[37.75,5.25
],
[21.75,12.25
],
[15.25,22.75]])
#print(pts)
#np.savetxt("pts.txt",pts)
#distance_matrix = sp.spatial.distance_matrix(pts, pts)
plt.xlim(0,60)
plt.ylim(0,60)
plt.imshow(image)


plt.plot(640, 570, "og", markersize=10)  # og:shorthand for green circle
plt.scatter(pts[:, 0], pts[:, 1], marker="x", color="red", s=200)
plt.savefig("vis.png")
plt.show()  
