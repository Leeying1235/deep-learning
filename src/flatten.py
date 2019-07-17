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
