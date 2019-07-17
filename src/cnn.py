import numpy as np
from layer import Layer
class ConvolutionLayer(Layer):
	'''
	For two demensional convolution
	'''
	def __init__(self,filters,kernel,stride,pad,activator):
		super(ConvolutionLayer,self).__init__()
		self.filters=filters
		#kernel:[kernel_h,kernel_w],values of kernel size 
		#should be odd,in order to make padding simple
		self.kernel=kernel
		#stride:[stride_h,stride_w]
		self.stride=stride
		#padding:'same','valid'
		self.pad=pad
		self.activator=activator
		self.__kernels=np.random.randn(self.filters,self.kernel[0],self.kernel[1])
		self.__bias=np.random.randn(self.filters)
	def get_patch(self,data,i,j):
		si=i*self.stride[0]
		sj=j*self.stride[1]
		return data[:,si:si+self.kernel[0],sj:sj+self.kernel[1]]
	def conv2D(self,_input,kernel,output,bi):
		[oh,ow]=output.shape
		for i in range(oh):
			for j in range(ow):
				output[i,j]=np.sum(self.get_patch(_input,i,j)*kernel)+bi
	def padding(self,_input):
		[m,ic,ih,iw]=_input.shape
		pad_h=(self.kernel[0]-1)/2
		pad_w=(self.kernel[1]-1)/2
		if self.pad=='same':
			pad_output=np.zeros((m,ic,ih+pad_h*2,iw+pad_w*2))
			pad_output[:,:,pad_h:pad_h+ih,pad_w:pad_w+iw]=_input
			return pad_output
		elif self.pad=='valid':
			return _input
		else:
			raise Exception('Only \'same\' and \'valid\' are implemented.')
	def forward(self,_input):
		#m:number of input samples
		#ic:input channel
		#ih:input height
		#iw:input width
		[m,ic,ih,iw]=_input.shape
		self.__input=self.padding(_input)
		[pm,pc,ph,pw]=self.__input.shape
		#calculate output size
		oh=(ph-self.kernel[0])//self.stride[0]+1
		ow=(pw-self.kernel[1])//self.stride[1]+1
		self.__unactivate_output=np.zeros((m,self.filters,oh,ow))
		#forward propagation
		for item in range(m):
			for nf in range(self.filters):
				self.conv2D(self.__input[item,:,:,:],self.__kernels[nf],self.__unactivate_output[item,nf,:,:],self.__bias[nf])
		return self._activation(self.activator,self.__unactivate_output)
	def flip_w(self,w):
		return np.fliplr(np.flipud(w))
	def expand(self,feature_map):
		[m,fc,fh,fw]=feature_map.shape
		[pm,pc,ph,pw]=self.__input.shape
		eh=ph-self.kernel[0]+1
		ew=pw-self.kernel[1]+1
		expand_feature=np.zeros((m,fc,eh,ew))
		for i in range(fh):
			for j in range(fw):
				expand_feature[:,:,i*self.stride[0],j*self.stride[1]]=feature_map[:,:,i,j]
		return expand_feature
	def back(self,error,learning_rate):
		#calculate delta^(l)
		error=error*self._activation_prime(self.activator,self.__unactivate_output)
		#back to stride[1,1]
		error=self.expand(error)
		[em,echannel,eh,ew]=error.shape
		#calculate gradient
		gradient_w=np.zeros((self.filters,self.kernel[0],self.kernel[1]))
		gradient_b=np.zeros(self.filters)
		for n in range(self.filters):
			for i in range(self.kernel[0]):
				for j in range(self.kernel[1]):
					gradient_w[n,i,j]=np.sum(self.__input[:,:,i:i+eh,j:j+eh]*error[:,n,:,:])
			gradient_b[n]=np.sum(error[:,n,:,:])
		#calculate back error
		[im,ichannel,ih,iw]=self.__input.shape
		back_error=np.zeros(self.__input.shape)
		#padding inorder to get same size of input array
		error=self.padding(error)
		pad_error=self.padding(error)
		sh=self.stride[0]
		sw=self.stride[1]
		self.stride[0]=self.stride[1]=1
		for item in range(em):
			for ic in range(ichannel):
				for ec in range(echannel):
					self.conv2D(pad_error[item,:,:,:],self.flip_w(self.__kernels[ec,:,:]),back_error[item,ic,:,:],0)
		self.stride[0]=sh
		self.stride[1]=sw
		#update
		self.__kernels-=learning_rate*gradient_w
		self.__bias-=learning_rate*gradient_b
		return back_error
