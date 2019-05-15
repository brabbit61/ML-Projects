import numpy as np
import math
def convertlabels(labels):
	x=[]
	for i in range(len(labels)):
		y=[0]*10
		y[labels[i]]=1
		x.append(y)
	return x
	
def normalize(x):
	n=len(x)
	for i in range(n):
		x[i]=int(x[i])
	s=sum(x)
	avg=s/n
	avg=np.tile(avg,(1,n))
	x=np.mat(x)
	diff=np.multiply((x-avg),(x-avg))
	var=np.sum(diff)/n
	x=(x-avg)/math.sqrt(var+0.000001)
	return x
	
def relu(x):
	return np.maximum(0,x)

def relu_prime(x):
	n=len(x)
	y=[0]*n
	for i in range(n):
		if x[i]>0:
			y[i]=1
	return np.mat(y)
	
def softmax(x):
	s=0
	n=len(x)
	y=[]
	for i in range(n):
		s+=np.exp(x[i])
	
	j=np.exp(x[0])
	j/=s
	y=np.mat(j)
	for i in range(1,n):
		j=np.exp(x[i])
		y=np.vstack((y,(j/s)))
	return np.mat(y)

info=[]
truelabels=[]

with open("/home/jenit1/Desktop/Assignments/Week5/train.csv") as fr:
	for x in fr.readlines():
		x=x.strip().split(",")
		truelabels.append(int(x[0]))
		info.append(x[1:])						#grayscale image of a 28*28 picture hence there are 784 pixels for 1 image
		if len(info)>5000:
			break

data=normalize(info[0])
for x in range(1,len(info)):
	data=np.vstack((data,normalize(info[x])))
m=len(info)
labels=convertlabels(truelabels)
labels=np.mat(labels)
x=np.ones((m,1))
x=np.hstack((x,data))
one=np.ones((1,1))
alpha=0.01
theta1=np.random.rand(25,785)/np.sqrt(25/2)					#25*785
theta2=np.random.rand(10,26)/np.sqrt(5)					#10*26
DELTA1=np.zeros((25,785))						#25*785
DELTA2=np.zeros((10,26))	
for i in range(700):
	print(i)
	for j in range(len(data)):
		a1=x[j]									#1*785
		z2=np.dot(theta1,a1.transpose())		#25*1
		a2=relu(z2)								#25*1
		aa2=np.vstack((one,a2))					#26*1
		z3=np.dot(theta2,aa2)					#10*1
		a3=softmax(z3)
		#backpropogation
		delta3=2*(a3-labels[j].T)					#10*1
		temp1=np.dot(theta2[:,:25].transpose(),delta3)	#25*1
		temp2=relu_prime(z2)		#25*1	
		delta2=np.multiply(temp1,temp2.transpose())			#25*1
		DELTA1=np.dot(delta2,a1)				#25*785
		DELTA2=np.dot(delta3,aa2.transpose())	
		alph=alpha/(i+j+1)
		theta1-=((alph/m)*DELTA1)
		theta2-=((alph/m)*DELTA2)		
	
	
error=0
for i in  range(len(truelabels)):
	a1=x[i]									#1*785
	z2=np.dot(theta1,a1.transpose())		#25*1
	a2=relu(z2)								#25*1
	aa2=np.vstack((one,a2))					#26*1
	z3=np.dot(theta2,aa2)					#10*1
	a3=softmax(z3)
	mp=np.max(a3)							#max probability
	index=0
	for j in np.array(a3).flat:
		if j==mp:
				break
		index+=1
	if index!=truelabels[i]:
		error+=1
print(error)