'''
from layer import Layer
class Flatten(Layer):
	def __init__(self):
		super(Flatten,self).__init__()
	def forward(self,_input):
		self.__input=_input
		self.__shape=self.__input.shape
		self.__output=self.__input.flatten()
		return self.__output
	def back(self,error,leraning_rate):
		return self.__output.reshape(self.__shape) 
'''
import numpy as np
from layer import Layer
class Flatten(Layer):
	def __init__(self):
		super(Flatten,self).__init__()
	def forward_propagation(self,_input):
		self.__input=_input
		self.__shape=self.__input.shape
		#self.__shape[0]:number of samples
		self.__output=np.zeros((self.__shape[0],reduce(lambda x,y:x*y,self.__shape[1:])))
		for i in range(self.__shape[0]):
			self.__output[i,:]=self.__input[i,:,:,:].flatten()
		return self.__output
	def back_propagation(self,error,leraning_rate):
		return self.__output.flatten().reshape(self.__shape) 
