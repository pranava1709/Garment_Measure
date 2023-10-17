import  pandas as pd
from matplotlib import pyplot as plt
import numpy as np

res = pd.read_csv('/content/drive/MyDrive/neww/128B_results/finalx.csv')
res1 =  pd.read_csv('/content/drive/MyDrive/neww/128B_results/finaly.csv')
c = 0
res3 = pd.read_csv('/content/drive/MyDrive/neww/128B_results/coord5.csv')
lst = []
lstc = []
lstd = []
lsti = []
cc = 0
for jj in res3.iloc[:,1:2].values:
  #print(jj)
  for ii in range(0,len(res1.values)):
    if jj == res3.iloc[:,1:2].values[ii]:
      c+=1
      if c>1:
        yy = jj
        xx = res3.iloc[ii:ii+1,0:1]
        #uu = pd.DataFrame(yy)
        #ww = pd.DataFrame(xx)
        lst.append(jj)
        lstc.append(xx)
print(lst)
pp = np.array(lst)
print(pp[0][0])
dd = np.array(lstc)
print(dd[0][0][0])
print(len(dd))
print(len(pp))

for q in range(0,len(pp)-1):
  sub = abs(dd[q+1][0][0] - dd[q][0][0])

  if sub < 27:
    continue
  lstd.append(sub)

ff = np.array(lstd)
ggg = np.sort(ff)
hhh = np.unique(ggg)
plt.xlabel("Set of Keypoints taken")
plt.ylabel("Distances")
print(hhh)
plt.scatter([tt for tt in range(0,7)],hhh)
plt.show()

print(res)
print(res1)
plt.xlabel('points on x-axis')
plt.ylabel('points on y-axis')
plt.scatter(res.values,res1.values)
plt.show()