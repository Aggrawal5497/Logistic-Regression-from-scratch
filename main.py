import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

#sigmoid function
def sigmoid(z):
	return (1.0 / (1.0 + np.exp(-z)))

#The function converts the results to binary values 0 and 1
def correct_data(data_val):
	data_val.replace({'Abnormal' : 0, 'Normal' : 1}, inplace = True)
	return data_val

#the cost function for Logistic regression
def cost_function(x , y, theta_val, size):
	y = y.reshape((-1, 1))
	val = np.matmul(x, theta_val).reshape(-1, 1)
	htheta = sigmoid(val)
	temp = -np.average(np.add(np.multiply(y, np.log(htheta)), np.multiply((1 - y), np.log(1 - htheta))))
	return temp

def gradient_decent(x, y, theta_val, size, learning_rate = 0.001):
	y = y.values.reshape((-1, 1))
	htheta = sigmoid(np.matmul(x, theta_val).reshape((-1, 1)))
	dif = htheta - y
	prod = (np.matmul(x.T, dif))
	theta_val = theta_val - (learning_rate * prod) / size
	return theta_val

def error_function(x, y, theta_val, size):
	y = y.reshape((-1, 1))
	htheta = get_answers(x, y, theta_val).reshape(-1, 1)
	dif = ((htheta - y) ** 2)
	return sum(dif) / size

#Converts float answers to binary classes
def get_answers(x_test, y_test, theta_val):
	ans = sigmoid(np.matmul(x_test, theta_val))
	for i in range(len(ans)):
		if ans[i] < 0.5:
			ans[i] = 0
		else:
			ans[i] = 1
	return ans.reshape(-1, 1)

#The function calculates the result
def result_calc(x, theta_val):
	return(sigmoid(np.matmul(x, theta_val).reshape((-1, 1))))

if __name__ == "__main__":
	data = pd.read_csv("Dataset_spine.csv") #load dataset
	data = correct_data(data)
	val = data.shape 
	train_size = int(val[0] * 0.8) #getting 80% of data for training
	ones = pd.DataFrame(np.ones((val[0], 1)))
	full_data = pd.concat([ones, data], axis = 1) #Adding biased value
	full_data = full_data.reindex(np.random.permutation(full_data.index)) #shuffling the dataset to avoid training on biased dataset
	theta = np.random.randn((val[1])).reshape((-1, 1))
	"""
	Spliting dataset into train and test sets
	"""
	train_x = full_data.loc[:train_size, :'scoliosis_slope']
	train_y = full_data.loc[:train_size, 'Result']
	test_x = full_data.loc[train_size + 1:, :'scoliosis_slope']
	test_y = full_data.loc[train_size + 1:, 'Result']
	for i in range(10000):
		theta = gradient_decent(train_x, train_y, theta, train_size)
		if (i+1) % 500 == 0:
			val = cost_function(train_x, train_y, theta, train_size)
			print("Cost after {} steps : {}".format(i+1,val))
	
	test_y = test_y.values.reshape((-1, 1))
	
	ans = sigmoid(np.matmul(test_x, theta))
	print("Accuracy: ",accuracy_score(test_y, get_answers(test_x, test_y, theta)))