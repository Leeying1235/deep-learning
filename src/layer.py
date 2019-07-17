import numpy as np
#activator
def sigmoid(x):
	#This activator may cause gradient disappear,and ease to be saturated,
	#and the output of this function is always positive.
	return 1.0/(1.0+np.exp(-x))
def relu(x):
	#Gradient of this activator always exsists,but the deal of the negative
	#value is too rough  
	return (abs(x)+x)/2.0
def tanh(x):
	#It's an improvement of sigmoid
	return np.tanh(x)
def softmax(x):
	x=x-np.max(x)
	return np.exp(x)/np.sum(np.exp(x))
def same(x):
	return x
#derivative
def sigmoid_prime(x):
	return sigmoid(x)*(1-sigmoid(x))
def relu_prime(x):
	return np.where(x>0,1,0)
def tanh_prime(x):
	return 1-tanh(x)*tanh(x)
def softmax_prime(x):
	len_x=len(x)
	result=np.zeros((len_x,len_x))
	for i in range(len_x):
		for j in range(len_x):
			if i==j:
				result[i,j]=x[i]*(1-x[i])
			else:
				result[i,j]=-x[i]*x[j]
	return result
def same_prime(x):
	return np.ones(x.shape)
#Base API
class Layer(object):
	def _activation(self,name,x):
		if name=='sigmoid':
			return sigmoid(x)
		elif name=='relu':
			return relu(x)
		elif name=='tanh':
			return tanh(x)
		elif name=='softmax':
			return softmax(x)
		elif name=='same':
			return same(x)
		else:
			raise Exception('None defined activator.')
	def _activation_prime(self,name,x):
		if name=='sigmoid':
			return sigmoid_prime(x)
		elif name=='relu':
			return relu_prime(x)
		elif name=='tanh':
			return tanh_prime(x)
		elif name=='softmax':
			return softmax_prime(x)
		elif name=='same':
			return same_prime(x)
		else:
			raise Exception('None defined activate derivative.')
	def forward(self,**kwargs):
		pass
	def back(self,**kwargs):
		pass
