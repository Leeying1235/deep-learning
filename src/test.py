from dense import DenseLayer
import model
import numpy as np

if __name__ =="__main__":
	model=model.Model()
	x=np.array([
	[1,1],
	[1,0],
	[0,1],
	[0,0]
	])
	y=np.array([
	[0],
	[1],
	[1],
	[0]
	])
	model.add(DenseLayer((2,2),'relu'))
	model.add(DenseLayer((2,4),'relu'))
	model.add(DenseLayer((4,1),'sigmoid'))
	model.compile("mse")
	model.fit(x,y,0.1,4,2000)
	#model.printm()
	print model.predict(np.array([[1,1],[0,1],[1,0],[0,0]]))
