import numpy as np

class Model(object):
	def __init__(self):
		self.layers=[]
	def add(self,layer):
		self.layers.append(layer)
	def compile(self,loss):
		if loss=="mse":
			self.__loss_function=self.__mse
		elif lose=="cross_entropy":
			self.__loss_function=self.__cross_entropy
		else:
			raise Exception("None difined loss function.")			
	def fit(self,X,y,learning_rate,batch_size,epochs):
		assert(len(X)==len(y))
		for i in range(epochs):
			lx=len(X)
			loss=0
			for b_index in range(lx//batch_size):
				index_temp=np.array(range(batch_size*b_index,(b_index+1)*batch_size))%lx
				out=X[index_temp]
				y_temp=y[index_temp]
				for layer in self.layers:
					out=layer.forward(out)
				loss+=self.__loss_function(out,y_temp,True)
				error=self.__loss_function(out,y_temp,False)
				for j in range(len(self.layers)-1):
					index=len(self.layers)-j-1
					error=self.layers[index].back(error,learning_rate)
			#print("epochs {} / {}  loss : {}".format(i + 1, epochs, loss/len(X)))
				
	def __mse(self,out,y,forward):
		if forward:
			return np.sum(np.sum(np.squeeze(0.5*(out-y)**2)))
		else:
			return out-y
	def __cross_entropy(self):
		return None
	def predict(self,X):
		res=[]
		for index in range(len(X)):
			out=X[index]
			for layer in self.layers:
				out=layer.forward(out)
			res.append(out)
		return np.squeeze(np.array(res))
