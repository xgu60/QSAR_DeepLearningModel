import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
	''' a simple dnn
	arguments: n_features e.g 1000
	           n_output e.g 1
	'''

	def __init__(self, n_features, n_output):
		super(DNN, self).__init__()
		self.h1 = nn.Linear(n_features, 4000)
		nn.init.xavier_uniform(self.h1.weight)
		self.h1_bn = nn.BatchNorm1d(4000)
		self.h1_relu = nn.ReLU()
		self.h1_dropout = nn.Dropout(p=0.75)
		self.h2 = nn.Linear(4000, 2000)
		nn.init.xavier_uniform(self.h2.weight)
		self.h2_bn = nn.BatchNorm1d(2000)
		self.h2_relu = nn.ReLU()
		self.h2_dropout = nn.Dropout(p=0.75)
		self.h3 = nn.Linear(2000, 1000)
		nn.init.xavier_uniform(self.h3.weight)
		self.h3_bn = nn.BatchNorm1d(1000)
		self.h3_relu = nn.ReLU()
		self.h3_dropout = nn.Dropout(p=0.50)						
		self.output = nn.Linear(1000, n_output)

	def forward(self, x):
		x = self.h1(x)
		x = self.h1_bn(x)
		x = self.h1_relu(x)
		x = self.h1_dropout(x)
		x = self.h2(x)
		x = self.h2_bn(x)
		x = self.h2_relu(x)
		x = self.h2_dropout(x)
		x = self.h3(x)
		x = self.h3_bn(x)
		x = self.h3_relu(x)	
		x = self.h3_dropout(x)					
		return self.output(x)



if __name__ == "__main__":
	net = DNN(777, 10)	
	print(net)
	

