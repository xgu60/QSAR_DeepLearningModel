import pandas as pd
import torch
from torch.utils.data.dataset import Dataset



class QSAR_Dataset(Dataset):
	'''read data from CSV file
	   Argument: CSV file name
	'''
	def __init__(self, dataframe):
		xy = dataframe		
		self.len = xy.shape[0]	
		self.x_data = torch.from_numpy(xy.iloc[:, 2:].as_matrix()).float()
		self.y_data = torch.from_numpy(xy.iloc[:, 1:2].as_matrix()).float()

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len

def getValidationData(dataframe):
	xy = dataframe
	x_data = torch.from_numpy(xy.iloc[:, 2:].as_matrix()).float()
	y_data = torch.from_numpy(xy.iloc[:, 1:2].as_matrix()).float()
	return x_data, y_data

def getTestData(dataframe):	
	data = torch.from_numpy(dataframe.iloc[:, 2:].as_matrix()).float()	
	smiles = dataframe.iloc[:, 1:2]
	return smiles, data




