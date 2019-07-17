import numpy as np
from layer import Layer
class DropoutLayer(Layer)
	def __init__(self,drop_ratio):
		super(DropoutLayer,self).__init__()
		self.__ratio=drop_ratio
	def forward_propagation(self,_input):
		self.__input=_input
		self.__mask=np.zeros(self.__input.shape)
		rand=np.random.rand(self.__input.shape)
		self.__mask=np.where(rand>self.__ratio,1,0)
		self.__output=(self.__input*self.__mask)/(1-self.__ratio)
		return self.__output
	def back_propagation(self,error,learning_rate):
		error=self.__mask*error
		error=error/(1-self.__ratio)
		return error
