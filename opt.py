import ast 
import pandas as pd
import numpy as np
lst_x= []
lst_y = []
lst_coord = []
lst_pos = []
opt_coord = []
lst_a = []
lst_x4 = []
lst_arr= []
lst_pos1 = []

x = pd.read_csv("finalx.csv")
y = pd.read_csv("finaly.csv")
x3 = x.iloc[:,0]
y3 = y.iloc[:,0]
for ia in x3:
	lst_x.append(ia)
#print(lst_x)
for ja in y3:
	lst_y.append(ja)
#print(lst_y)
for r,s in zip(lst_x,lst_y):
	coor = str(r)+str(",")+str(s)
	print(coor)
	lst_coord.append(coor)
df = pd.DataFrame(lst_coord)
df.to_csv("coordinates.csv")
coordinates= pd.read_csv("coordinates.csv")
coordinates = coordinates.iloc[1:,1]
coordinates= coordinates.values
#print(coordinates)
c = 0
loc = 0
locy = 0
for y in coordinates:
	c1 = np.char.count(str(coordinates),str(y))
	y = np.array(y)
	if c1>1:
		for s,v in np.ndenumerate(coordinates):
			if v==y:
				#print(coordinates)
				#lst_pos.append(s)
				#print(lst_pos[1])
				df1 = pd.DataFrame(s)
				for a in s:
					print("DUPLICATES")

		#print(type(a))
		lst_a.append(a)

		a1 =pd.DataFrame(lst_a)
		print(len(a1))
		a1 = a1.iloc[0:len(a1)//3,:]
		#print(a1.shape)
		x3 = coordinates[a1.iloc[:,0]]
		x3 = pd.DataFrame(x3)
		x3.to_csv("opt"+str(c)+'.csv',index = False)
		#c = c+1
		x3a = pd.read_csv("opt0.csv")
		x3a = pd.DataFrame(x3)

		
	else:
		x4 = y
		print(x4)
		lst_x4.append(x4)
		a2 = pd.DataFrame(lst_x4)
		#print(a2)
		#with open("fin4.txt",'a') as z:
		#	z.write(str(x4))
		#	z.write('\n')



		
#print(x3)
with open("fin4.txt",'a') as z:
	z.write(str(x3a))
	z.write('\n')


with open("fin4.txt",'r') as n:
	file = n.readlines()
	#print(file)	
	final_df = pd.DataFrame(file)
	final_df.columns= ['COORD']
	final_df = final_df.COORD.str.split(expand = True)
	index3 = final_df.iloc[23:41,1]
	index4  = final_df.iloc[0:22,0]
	#final_df = final_df.drop(index4)
	
	#print(index3)
	#print(index4)
	fin_df = pd.concat([index4,index3],axis=0)

	
	fin_arr = np.array(fin_df)
	for u in fin_arr:
		print(u)
		fin_arr1 = str(u).replace(',',' ')
		print(fin_arr1)
		lst_arr.append(fin_arr1)
		#fin_arr2.to_csv("finarr.csv")
fin_arrdf = pd.DataFrame(lst_arr)
print(fin_arrdf)
fin_arrdf.columns = ['coordinate']
fin_arrdf1 = fin_arrdf.coordinate.str.split(expand = True)

	
	
print(fin_arrdf1)
fin_arr_df2 = pd.concat([fin_arrdf1,fin_df],axis=1)
print(fin_arr_df2)
xx = fin_arr_df2.iloc[:,0]
yy = fin_arr_df2.iloc[:,1]
xxyy = fin_arr_df2.iloc[:,2]
xxyy = np.array(xxyy)
print(xxyy)

for r in xx:
	c2 = np.char.count(str(xx),str(r))
	#print(c2)
	if c2>1:
		for ss,vv in np.ndenumerate(xx):
			if vv==r:
				#print(ss)
				for jj in ss:
					
					lst_pos1.append(jj)
					#print(lst_pos1)
				
jjdf = pd.DataFrame(lst_pos1)
print(jjdf)


#for h in range(0,len(jjdf)+1):
jjdf1 = jjdf.iloc[0:1,:]
jjdf2 = jjdf.iloc[1:2,:]
a1 = xxyy[jjdf1]
a2 = xxyy[jjdf2]
print(a1)
print(a2)

if a1 == None or a2 == None:
	print(a1.size)
	print(a2)
else:
	jjdf1a = [ast.literal_eval(d) for d in a1]
	print(jjdf1a)
	jjdf1b = [ast.literal_eval(e) for e in a2]
	print(jjdf1b)
	dist = np.subtract(jjdf1a,jjdf1b)
	print(dist)

	np.savetxt("distance_pix.txt",dist)	











		

		

	
	