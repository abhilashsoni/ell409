import numpy as np 
import matplotlib.pyplot as plt
from numpy import genfromtxt
from numpy import linalg as LA
import mnist_reader
import cv2

def separateByclass(x,y,k):
	X= []
	for i in range(0,k):
		t = x[y[:]==i,:]
		X.append(t)
	return X

def dataProcessing(d,k):
	N = np.size(d,axis=0)
	dtrain = d[0:2*N/3,:]
	dtest = d[2*N/3:,:]
	xtrain=dtrain[:,1:]
	ytrain=dtrain[:,0]
	xtest = dtest[:,1:]
	ytest = dtest[:,0]
	x = []
	for i in range(0,k):
		t = dtrain[dtrain[:,0]==i,:]
		x.append(t[:,1:])

	return [xtrain,ytrain,xtest,ytest,x]


def DataVisualization(x):
	plt.figure(0)
	marker=['o','+','^','x','D','*','h','8','p','s','|','_']
	for i in range(0,k):
		plt.scatter(x[i][:,0],x[i][:,1],marker=marker[i])
	plt.show()


def MLEGaussian(x):
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
	print ' *'
	t=np.sum(sigma,axis=0)
	det = np.product(t)
	print det
	z = np.dot(x-mu,sigmainv)
	z = np.dot(z,(x-mu).transpose())
	z = z * np.eye(n)
	z = np.sum(z,axis=0)
	z = np.exp(-1*z)
	z = z/(((2*3.14)**(d/2))*det)
	return z


	
def PCA(m,c):
	mean = np.mean(m,axis=0)
	m=m-mean
	cov = np.cov(m.transpose())
	eigval , eigvec = LA.eig(cov)
	a = np.argsort(eigval)
	b = eigvec[a]
	return b[0:c]



def visualizeMNIST(x,s):
	print np.size(x)
	s1=x.size**0.5
	s1=int(s1)
	y=np.reshape(x,(s1,s1))
	plt.imshow(y,cmap='gray')
	plt.savefig(s)

def classifyNaiiveBayes(x,xtest,ytest,N):
	theta = []
	cc = []
	prior = []
	posterior = []
	for i in range(0,k):
		t = MLEGaussian(x[i])
		ccond = Naiivegaussian(xtest,t[0],t[1])
		p = np.size(x[i],axis=0)*1.0/N
		pos = ccond*p
		theta.append(t)
		cc.append(ccond)
		prior.append(p)
		posterior.append(pos)

	bayes = np.argmax(posterior,axis=0)
	acc = calculateAccuracy(bayes,ytest)
	print "Accuracy on test set for naiive bayes classifier is ",acc

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
	# print np.shape(cov)
	sign,logdet = np.linalg.slogdet(cov)
	# print sign, " ",logdet
	logdet=logdet/2
	det = sign*np.exp(logdet)
	z = z/(((2*3.14)**(d/2))*det)
	return z

def MLE(x):
	mean = np.mean(x,axis=0)
	var = np.cov(x.transpose())
	# print np.size(mean)
	return [mean,var]

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

def kMeans(xtrain,km):
	theta = MLEGaussian(xtrain)
	var = theta[1]
	mu = theta[0]
	rnd = np.random.rand(km,np.size(xtrain,axis=1))
	centroid = var*rnd+mu
	
	c=0
	while(True):

		prev=np.copy(centroid)
		dist = distanceMetric(xtrain,centroid)
		curr = np.argmin(dist,axis=0)
		for i in range(0,km):
			temp=np.where(curr==i)
			x1=xtrain[temp]
			centroid[i]=np.mean(x1,axis=0)
		update=distanceMetric(prev,centroid)
		update=update*np.eye(km,km)
		update=np.sum(update,axis=0)
		mindiff=np.min(update)
		kMeansVisualisation(xtrain,centroid,km,c)
		c+=1
		print c
		print mindiff
		if(mindiff<=0.00000001):
			break;


	plt.clf()
	plt.scatter(xtrain[:,0],xtrain[:,1],color='grey',s=2)
	plt.savefig('unclustered.png')



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

def PCAVisualize(u,x):
	xtrain1=np.dot(x,u)
	xnew=np.dot(u,xtrain1)
	s='test'+'.png'
	visualizeMNIST(xnew,s)


k = 10 # Number of classes
xtrain, ytrain = mnist_reader.load_mnist('data/fashion', kind='train')
xtest, ytest = mnist_reader.load_mnist('data/fashion', kind='t10k')
N=np.size(xtrain,axis=0)
u = PCA(xtrain,3)
u=u.transpose()
xtrain1=np.dot(xtrain,u)
xtest1=np.dot(xtest,u)
x = separateByclass(xtrain1,ytrain,k)
print np.shape(xtrain1)
# classifyBayes(x,xtest1,ytest,N)
# classifyKNN(xtrain1,ytrain,xtest1,ytest,10)

PCAVisualize(u,xtrain[100])
visualizeMNIST(xtrain[100],"real.png")
# xtrain = np.dot()




