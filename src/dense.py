import numpy as np
from layer import Layer
class DenseLayer(Layer):
	def __init__(self,shape,activator):
		super(DenseLayer,self).__init__()
		#shape:[input_width,output_width]
		self.shape=shape
		self.activator=activator
		self.__w=2*np.random.randn(self.shape[0],self.shape[1])
		self.__b=np.random.randn(self.shape[1])
	def forward(self,_input):
		#z^(l)=a^(l-1)*w^(l)+b^(l)
		#z^(l):m;a^(l-1):n;b^(l):m;w^(l):nxm
		self.__input=_input #a^(l-1)
		self.__unactivate_output=np.dot(self.__input,self.__w)+self.__b#z^(l)
		#print self.__unactivate_output
		return self._activation(self.activator,self.__unactivate_output)#a^(l)
	def back(self,error,learning_rate):
		#delta^(L)=phi(J)/phi(z^(l))
		#calculate delta^(l)
		delta=error*self._activation_prime(self.activator,self.__unactivate_output)
		#calculate gradient
		gradient_w=np.dot(self.__input.T,delta)
		gradient_b=np.sum(delta.T,1)
		#calculate back error
		back_error=np.dot(delta,self.__w.T)
		#update
		self.__w-=learning_rate*gradient_w
		self.__b-=learning_rate*gradient_b
		return back_error
