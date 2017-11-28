import random
import numpy as np
import pickle as pkl

print "xor Deep Neural Network"
random.seed(1235)
# 레이어가 2개 만들었더니 초기 랜덤값에 영향을 너무 받아서 시드를 주었습니다.
# 이 시드값으로 돌렸을때 교수님커퓨터로 돌아가지 않는다면 시드를 제거하고 여러번 돌려주시면 감사하겠습니다 ㅠㅠㅠ
# 초기값이 운이 나쁠경우엔 여러번돌려도 결국 학습이 되지 않는 경우가 발생하여
# 시간낭비를 줄이고자 4개를 총 학습하는 횟루를 최대 1000번으로 지정하였습니다.
# 초기값이 잘 선택된경우네는 천번 이상 학습했을때야 값이 나오는 경우보다는 더 작은 경우가 많아서 그렇게 했습니다.

# [0,0]	[1,0] [0,1] [1,1]
dataSet = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
correctSet = np.array([0,1,1,0])

_N_ROW = 4
_N_COL = 2	

# logistic activation
def logistic(z):
    return (1.0 / (1.0 + np.exp(-z))).astype('float32')

# Relu function
def relu(z):
	for i in range(0,_N_ROW):
		for j in  range(0,_N_COL):
			if(z[i][j] < 0):
				z[i][j] = 0
	return z

def reluderi(z):
	for i in range(0,_N_ROW):
		for j in  range(0,_N_COL+1):
			if(z[i][j] > 0 ):
				z[i][j] = 1
	
	return z

def save(fn,obj):
    fd = open(fn,'wb')
    pkl.dump(obj,fd)
    fd.close() 


def MSP():
	# multi layer perceptron funcion
	# hidden layer 2, output layer 1
	# 	lnput 		h1 		h2 		output
	#	  O 		O  		O
	#	  O 		O 		O 			O
	#	  O 		O 		O
	fd = open("train_log.txt",'w')

	##############################################
	###############  train #######################
	##############################################

	# hidden layer weight 2 initalization
	hidden_weight = np.ones((12))
	hidden_weight = hidden_weight.reshape(2,3,2)
	for i in range(0,2):
		weight = np.array([[random.uniform(-1.0,1.0),random.uniform(-1.0,1.0),random.uniform(-1.0,1.0)],[random.uniform(-1.0,1.0),random.uniform(-1.0,1.0),random.uniform(-1.0,1.0)]]).astype('float32')
		weight = weight.reshape(3,2)

		hidden_weight[i] = weight
	print 'hidden_weigth init'
	print hidden_weight

	# output layer weigth initialization
	# output weigth => 1*3
	output_weight = np.array([random.uniform(-1.0,1.0),random.uniform(-1.0,1.0),random.uniform(-1.0,1.0)]).astype('float32')
	print 'output_weight init'
	print output_weight

	one_array = np.ones((4)).astype('float32')

	for i in range(0,1000):
		error = 0;

		hidden = np.ones((24)).reshape(3,4,2)
		layer_input = np.ones((3*4*3)).reshape(3,4,3)
		layer_input[0] = dataSet

		for j in range(0,2):
			# (4*3,3*2) => 4*2
			hidden[j] = relu(np.dot(layer_input[j],hidden_weight[j]))
			layer_input[j+1][:,[1,2]] = hidden[j]

		##### output layer
		# 4*3 3*1 => 4*1
		output = np.dot(layer_input[2],output_weight)
		#result 
		# activation function -> sigmoid fucntion
		y = logistic(output)

		print 'y->',y
		for j in range(0,4):
			# output >= 0.5 -> 1, output < 0.5 -> 0
			if(y[j] >= 0.5):
				if(correctSet[j] != 1):
					error = error + 1
			else:
				if(correctSet[j] != 0):
					error = error + 1

		print i,'th error number ->', error

		cost = 0.5 * np.dot((correctSet - y),(correctSet-y))
		fd.write('%d th cost -> %f\n' % (i,cost))
		fd.write('%d th error number ->%d\n' % (i,error))

		# all correct ---> stop
		if error == 0:
			print i," th stop (error = 0) --------------------"
			fd.write('1th hidden layer parameter -> [[%f,%f,%f],[%f,%f,%f]]\n'% (hidden_weight[0][0,0],hidden_weight[0][1,0],hidden_weight[0][2,0],hidden_weight[0][0,1],hidden_weight[0][1,1],hidden_weight[0][2,1]))
			fd.write('2th hidden layer parameter -> [[%f,%f,%f],[%f,%f,%f]]\n'% (hidden_weight[1][0,0],hidden_weight[1][1,0],hidden_weight[1][2,0],hidden_weight[1][0,1],hidden_weight[1][1,1],hidden_weight[1][2,1]))
			fd.write('output layer parameter -> [%f,%f,%f]\n'% (output_weight[0],output_weight[1],output_weight[2]))
			fd.close()
			break;	

		print 'cost---------->',cost

		# output sigmoid gradient 
		# (d-o)o(1-o)
		out_delta = np.multiply((correctSet-y),np.multiply(y,(one_array-y)))
		#### output layer weight update
		# w + delta*h
		output_weight = output_weight + np.dot(out_delta,layer_input[2])

		# hidden relu gradient compute
		hidden_grad = np.ones((2*4*3)).reshape(2,4,3)
		hidden_grad[1] = reluderi(layer_input[2])
		hidden_grad[0] = reluderi(layer_input[1])

		# hidden layer 2 delta2
		# delta * g'*w
 		hidden_delta= np.multiply(out_delta.reshape(4,1),np.multiply(hidden_grad[1],output_weight))

 		# w + delta1 *h
 		hidden_weight[1][1:,0] = hidden_weight[1][1:,0] - np.dot(hidden_delta[:,0],layer_input[1][:,1:]) 
 		hidden_weight[1][1:,1] = hidden_weight[1][1:,1] - np.dot(hidden_delta[:,1],layer_input[1][:,1:]) 

 		# hidden layer 1 delta1
		# delta1 * g'*w
 		hidden_delta_1= np.dot(np.multiply(hidden_grad[0],hidden_delta),hidden_weight[1])

 		# w + delta2 *h
 		hidden_weight[0][1:,0] = hidden_weight[0][1:,0] - np.dot(hidden_delta_1[:,0],layer_input[0][:,1:]) 
 		hidden_weight[0][1:,1] = hidden_weight[0][1:,1] - np.dot(hidden_delta_1[:,1],layer_input[0][:,1:]) 

	fd.close()

	##############################################
	##################  test    ##################
	##############################################
	bias = np.ones((4)).reshape(4,1)

	h1 = relu(np.dot(dataSet,hidden_weight[0]))
	hh1 = np.concatenate((bias,h1),axis=1)

	h2 = relu(np.dot(hh1,hidden_weight[1]))
	hh2 = np.concatenate((bias,h2),axis=1)

	output = logistic(np.dot(hh2,output_weight))
	error = 0

	fd = open("test_output.txt",'w')

	print "XOR Test->"

	for j in range(0,4):
		if(output[j] >= 0.5):
			temp = 1
		else:
			temp = 0
		if(correctSet[j] != temp):
			fd.write('y --> %d correct --> %d ==> false\n' % (temp,correctSet[j]))
			print "y-->",temp,"correct -->",correctSet[j],"==> false"
		else:
			fd.write('y --> %d correct --> %d ==> correct!!\n'% (temp,correctSet[j]))
			print "y-->",temp,"correct -->",correctSet[j],"==> correct"

	print 'error # ->',error
	fd.write('error # -> %d' % (error))
	fd.close()

	return;

MSP()