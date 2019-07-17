import numpy as np
from layer import Layer
class MaxPoolingLayer(Layer):
	def __init__(self,p_shape,stride,name):
		super(MaxPoolingLayer,self).__init__()
		self.__p_h=p_shape[0]
		self.__p_w=p_shape[1]
		self.__stride_h=stride[0]
		self.__stride_w=stride[1]
		self.__name=name
		self.__input=None
		self.__output=None
	def forward_propagation(self,_input):
		input_shape=_input.shape
		self.__input=_input
		self.__channel=_input.shape[0]
		self.__out_h=input_shape[1]/self.__p_h
		self.__out_w=input_shape[2]/self.__p_w
		self.__mask=np.zeros(input_shape)
		self.__output=np.zeros((self.__channel,self.__out_h,self.__out_w))
		for c in range(self.__channel):
			for i in range(self.__out_h):
				for j in range(self.__out_w):
					s_i=i*self.__stride_h
					s_j=j*self.__stride_w
					arr=self.get_patch(c,s_i,s_j)
					_m=np.max(arr)
					self.__output[c,i,j]=_m
					[m_i,m_j]=self.get_index(arr,_m)
					self.__mask[c,s_i+m_i,s_j+m_j]=1
		return self.__output
	def back_propagation(self,error,learning_rate):
		error=self.upsample(error)
		return error*self.__mask
	def upsample(self,error):
		shape=error.shape
		up_error=np.ones(self.__input.shape)
		for c in range(shape[0]):
			for i in range(shape[1]):
				for j in range(shape[2]):
					s_i=i*self.__stride_h
					s_j=j*self.__stride_w
					up_error[c,s_i:s_i+self.__p_h,s_j:s_j+self.__p_w]*=error[c,i,j]
		return up_error
	def get_patch(self,c,start_i,start_j):
		return self.__input[c,start_i:start_i+self.__p_h,start_j:start_j+self.__p_w]
	def get_index(self,arr,t):
		shape=arr.shape
		for i in range(shape[0]):
			for j in range(shape[1]):
				if arr[i,j]==t:
					return [i,j]
		return [0,0]

class MeanPoolingLayer(Layer):
	def __init__(self,p_shape,stride,name):
		super(MeanPoolingLayer,self).__init__()
		#pooling shape
		self.__p_h=p_shape[0]
		self.__p_w=p_shape[1]
		self.__stride_h=stride[0]
		self.__stride_w=stride[1]
		self.__name=name
		self.__input=None
		self.__output=None
	def forward_propagation(self,_input):
		input_shape=_input.shape
		self.__input=_input
		self.__channel=_input.shape[0]
		self.__out_h=input_shape[1]/self.__p_h
		self.__out_w=input_shape[2]/self.__p_w
		self.__output=np.zeros((self.__channel,self.__out_h,self.__out_w))
		for c in range(self.__channel):
			for i in range(self.__out_h):
				for j in range(self.__out_w):
					s_i=i*self.__stride_h
					s_j=j*self.__stride_w
					self.__output[c,i,j]=np.average(self.get_patch(c,s_i,s_j))
		return self.__output
	def back_propagation(self,error,learning_rate):
		error=self.upsample(error)/(self.__p_w*self.__p_h*1.0)
		return error
	def upsample(self,error):
		shape=error.shape
		up_error=np.ones(self.__input.shape)
		for c in range(shape[0]):
			for i in range(shape[1]):
				for j in range(shape[2]):
					s_i=i*self.__stride_h
					s_j=j*self.__stride_w
					up_error[c,s_i:s_i+self.__p_h,s_j:s_j+self.__p_w]*=error[c,i,j]
		return up_error
	def get_patch(self,c,start_i,start_j):
		return self.__input[c,start_i:start_i+self.__p_h,start_j:start_j+self.__p_w]
	def get_index(self,arr,t):
		shape=arr.shape
		for i in range(shape[1]):
			for j in range(shape[2]):
				if arr[c,i,j]==t:
					return [i,j]
		return [0,0]
