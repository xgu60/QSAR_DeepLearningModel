from data_loader import getTestData
from train import train_model
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd

def predict(folder, filename, savePoints):		
	NET_SAVE_POINTS = savePoints
	PREDS = []	
	folderpath = folder
	FILE = filename
	

	smiles, data = getTestData(folderpath + FILE)
	
	
	data = Variable(data.cuda())
	

	for epoch in NET_SAVE_POINTS:
		net = torch.load(folderpath + 'net/epoch' + str(epoch) + '.pkl')
		net.eval()
		pred = net(data)				
		PREDS.append(pred)	
		

	#average the prediction
	pred_sum = torch.cat(PREDS, 1)
	pred_mean = torch.mean(pred_sum, 1, True)
	
	#convert data to numpy array for further analysis
	pred = pred_mean.data.cpu().numpy()	
	res = np.column_stack((smiles, pred))

	return res

if __name__ == "__main__":	
		
	res = predict("vs/", "data/smiles_0/1th.csv", [40, 45, 50])
	
	df = pd.DataFrame(res)
	df.to_csv("predictions.csv")
