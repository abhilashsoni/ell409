import numpy as np 
import matplotlib.pyplot as plt
from numpy import genfromtxt
from numpy import linalg as LA
import random
def separateByclass(x,y,k):
	X= []
	for i in range(0,k):
		t = x[y[:]==i,:]
		X.append(t)
	return X


	
def PCA(m,k):
	mean = np.mean(m,axis=0)
	m=m-mean
	cov = np.cov(m.transpose())
	eigval , eigvec = LA.eig(cov)
	a = np.argsort(eigval)
	b = eigvec[a]
	comp = b[0:k]
	comp = comp.transpose()
	xtrain1=np.dot(m,comp)
	return xtrain1

def DataVisualization(x):
	plt.figure(0)
	marker=['o','+','^','x','D','*','h','8','p','s','|','_']
	for i in range(0,k):
		plt.scatter(x[i][:,0],x[i][:,1],marker=marker[i])
	plt.show()

def MLE(x):
	mean = np.mean(x,axis=0)
	var = np.cov(x.transpose())
	print np.size(mean)
	return [mean,var]

def BayesianEstimate(mu,cov,n):
	d = np.size(cov,axis=1)
	mu0 = np.zeros((1,d))
	sigma = cov
	sigma0 = np.eye(d,d)
	t = sigma0+sigma*1.0/n
	t = np.linalg.inv(t)
	t1 = np.dot(sigma0,t)
	t2 = np.dot(sigma,t)
	t1 = np.dot(mu,t1)
	t2 = np.dot(mu0,t2)
	mean = t1+t2
	sigman = np.dot(t1,sigma)
	sigman = sigman*1.0/n
	sigman = sigma+sigman
	return [mean,sigman]


def MLENaiive(x):
	mean = np.mean(x,axis=0)
	var = np.var(x,axis=0)
	return [mean,var]


def distanceMetric(x1,x2):
	x1=np.asarray(x1)
	x2=np.asarray(x2)
	m=np.size(x1,axis=0)
	n=np.size(x2,axis=0)
	d=np.size(x1,axis=1)
	x1=np.reshape(x1,(1,m,d))
	x2=np.reshape(x2,(n,1,d))
	diff = x1-x2
	diff=diff*diff
	res = np.sum(diff,axis=2)
	return res # every row corresponds to different test sample, every column represents training sample

def calculateAccuracy(ycalc,ytest):
	err = ycalc-ytest
	err = np.where(err==0)
	acc = np.size(err)*1.0/np.size(ycalc)*100
	return acc


def Naiivegaussian(x,mu,var):
	d = np.size(x,axis=1)
	n = np.size(x,axis=0)
	sig = np.eye((d))
	sigma = sig*var
	sigmainv=sig/var
	det = np.product(np.sum(sigma,axis=0))
	z = np.dot(x-mu,sigmainv)
	z = np.dot(z,(x-mu).transpose())
	z = z * np.eye(n)
	z = np.sum(z,axis=0)
	z = np.exp(-1*z)
	z = z/(((2*3.14)**(d/2))*(det**(1/2)))
	return z

def gaussian(x,mu,cov):
	d=np.size(x,axis=1)
	n=np.size(x,axis=0)
	sigma = cov
	sigmainv=np.linalg.inv(cov)
	x1 = x-mu
	t = np.dot(x1,sigmainv)
	t = np.dot(t,x1.transpose())
	z = t * np.eye(n)
	z = np.sum(z,axis=0)
	z = np.exp(-1*z)
	det = np.linalg.det(cov)
	z = z/(((2*3.14)**(d/2))*(det**(1/2)))
	return z


def classifyBayes(x,xtest,ytest,N):
	theta = []
	cc = []
	prior = []
	posterior = []
	for i in range(0,k):
		t = MLE(x[i])
		ccond = gaussian(xtest,t[0],t[1])
		p = np.size(x[i],axis=0)*1.0/N
		pos = ccond*p
		theta.append(t)
		cc.append(ccond)
		prior.append(p)
		posterior.append(pos)
	bayes = np.argmax(posterior,axis=0)
	acc = calculateAccuracy(bayes,ytest)
	print "Accuracy on test set for bayes classifier is ",acc
	return acc

def classifyBayesBayesianEstimate(x,xtest,ytest,N):
	theta = []
	cc = []
	prior = []
	posterior = []
	for i in range(0,k):
		t = MLE(x[i])
		m,v = BayesianEstimate(t[0],t[1],np.size(x[i],axis=0))
		ccond = gaussian(xtest,m,v)
		p = np.size(x[i],axis=0)*1.0/N
		pos = ccond*p
		theta.append(t)
		cc.append(ccond)
		prior.append(p)
		posterior.append(pos)
	bayes = np.argmax(posterior,axis=0)
	acc = calculateAccuracy(bayes,ytest)
	print "Accuracy on test set for bayes classifier is ",acc
	return acc



def classifyNaiiveBayes(x,xtest,ytest,N):
	theta = []
	cc = []
	prior = []
	posterior = []
	for i in range(0,k):
		t = MLENaiive(x[i])
		ccond = Naiivegaussian(xtest,t[0],t[1])
		p = np.size(x[i],axis=0)*1.0/N
		pos = ccond*p
		theta.append(t)
		cc.append(ccond)
		prior.append(p)
		posterior.append(pos)

	bayes = np.argmax(posterior,axis=0)
	acc = calculateAccuracy(bayes,ytest)
	# print "Accuracy on test set for naiive bayes classifier is ",acc
	return acc



def classifyKNN(xtrain,ytrain,xtest,ytest, knn):
	n = np.size(xtest,axis=0)
	res = distanceMetric(xtrain,xtest)
	order = np.argsort(res,axis=1)
	order = order[:,0:knn]
	y = np.reshape(ytrain,(np.size(ytrain),1))
	res=y[order]
	res=np.squeeze(res)
	ans=np.zeros((k,np.size(xtest,axis=0)))
	for i in range(0,k):
		f=np.sum((res==i)*1,axis=1)
		ans[i]=f

	ans=ans.transpose()
	ans=np.argmax(ans,axis=1)
	acc=calculateAccuracy(ans,ytest)
	print "Accuracy for k-nearest neighbour with k = ",knn," is ",acc
	return acc

def kMeans(xtrain,km,y,xtest,ytest):
	# theta = MLENaiive(xtrain)
	# print "STARTTTTT"
	# var = theta[1]
	# mu = theta[0]
	# print mu
	# rnd = np.random.rand(km,np.size(xtrain,axis=1))
	# centroid = var*rnd+mu
	N = np.size(xtrain,axis=0)
	number = random.sample(xrange(1,N), km)
	centroid = xtrain[number]
	# print "CENTROID INITIAL ",centroid
	c=0
	while(True):
		prev=np.copy(centroid)
		dist = distanceMetric(xtrain,centroid)
		print "Distane Metric ", np.shape(dist)
		curr = np.argmin(dist,axis=0)
		print np.shape(curr)
		print np.bincount(curr)
		X=[]
		for i in range(0,km):
			temp=np.where(curr==i)
			x1=xtrain[temp]
			n1=np.size(temp)
			if(n1>0):
				centroid[i]=np.mean(x1,axis=0)
				X.append(temp)
			else:
				X.append(temp)
		update=distanceMetric(prev,centroid)
		update=update*np.eye(km,km)
		update=np.sum(update,axis=0)
		# print update[0]
		mindiff=np.min(update)
		c+=1
		print mindiff
		if(mindiff<=0.0000001):
			break;
	cl =np.zeros((km,km))
	for i in range(0,km):
		t = X[i]
		p = y[t]
		p = p.astype(int)
		count = np.bincount(p)
		count = np.asarray(count)
		if(np.size(count)<km):
			count = np.append(count,0)
		count = count.reshape(1,np.size(count))
		cl[i]=count
	classes = np.argmax(cl,axis=1)
	M = np.size(xtest,axis=0)
	dist = distanceMetric(xtest,centroid)
	lab = np.argmin(dist,axis=0)
	y2 = np.zeros(M)
	for i in range(0,km):
		y2[np.where(lab==i)]=classes[i]
	y2=y2.astype(int)
	print "Kmeans accuracy " ,calculateAccuracy(y2,ytest)










	# plt.scatter(xtrain[:,0],xtrain[:,1],color='grey',s=2)
	# plt.savefig('unclustered.png')



	# print centroid


def kMeansVisualisation(xtrain,centroid,km,i1):
	plt.figure(1)
	dist = distanceMetric(xtrain,centroid)
	curr = np.argmin(dist,axis=0)
	color=['red','blue','green','yellow','orange','purple','black','pink','aqua']
	for i in range(0,km):
		temp=np.where(curr==i)
		x1=xtrain[temp]
		plt.scatter(centroid[i][0],centroid[i][1],s=100,color='black',marker='X')
		plt.scatter(x1[:,0],x1[:,1],s=2,color=color[i])
	name=str(i1)+'.png'
	print name
	plt.savefig(name)
	plt.clf()


def OptimumKNN(xtrain,ytrain,xtest,ytest):
	res=[]
	index=[]
	for i in range(k,100):
		t=classifyKNN(xtrain,ytrain,xtest,ytest,i)
		res.append(t)
		index.append(i)

	opt = np.argmax(res)
	print "Optimum K-nearest neighbour is at K=",opt+3," and gives accuracy ",res[opt]

	plt.figure(1)
	plt.plot(index,res)
	plt.show()



dtrain = genfromtxt('medicalData.txt')
dtest = genfromtxt('medicaltest.txt')
k = 3 # Number of classes
N = np.size(dtrain,axis=0)
xtrain=dtrain[:,1:]
ytrain=dtrain[:,0]
xtest = dtest[:,1:]
ytest = dtest[:,0]
x = separateByclass(xtrain,ytrain,k)
classifyBayes(x,xtest,ytest,N)


kMeans(xtrain,3,ytrain,xtest,ytest)
# OptimumKNN(xtrain,ytrain,xtest,ytest)
# xtrain1 = PCA(xtrain,2)
# xtest1 = PCA(xtest,2)
# x1 = separateByclass(xtrain1,ytrain,3)
# classifyNaiiveBayes(x1,xtest1,ytest,N)
# OptimumKNN(xtrain1,ytrain,xtest1,ytest)
# DataVisualization(x1)

