
import numpy as np
import pandas as pd
from preprocess import shuffle_dataframe
from train import train_model
from test import test_model

if __name__ == "__main__":
	df = pd.read_csv("pg/data/processed_data.csv")	
	netPath = "pg/net/"
	shuffle = True
	randomSeeds = range(0,10)
	train_pctl = 0.7
	train_vis = False
	epoch_num = 51
	batch_size = 64
	test_vis = False
	sp = [40, 45, 50]
	
	train_r2s = []
	val_r2s = []
	aucs = []
	for seed in randomSeeds:
		print("the random seed: {}".format(seed))
		train, val = shuffle_dataframe(df, shuffle=shuffle, seed=seed, pctl=train_pctl)
		train_model(train, val, visual=train_vis, epoch=epoch_num, batchSize=batch_size, 
					savePath=netPath, savePoints=sp)
		train_r2, val_r2, auc = test_model(train, val, visual=test_vis, netPath=netPath, 
											savePoints=sp)
		train_r2s.append(train_r2)
		val_r2s.append(val_r2)
		aucs.append(auc)
		# store res in temp file
		# res = {"train_r2" : train_r2s, 
			# "val_r2" : val_r2s, 
			# "auc" : aucs}
		# df = pd.DataFrame(res, columns=["train_r2", "val_r2", "auc"])
		# df.loc["mean"] = df.mean()
		# df.to_csv("exp_temp.csv")
	

	res = {"seeds": randomSeeds, 
			"train_r2" : train_r2s, 
			"val_r2" : val_r2s, 
			"auc" : aucs}
	df = pd.DataFrame(res, columns=["seeds", "train_r2", "val_r2", "auc"])
	df.loc["mean"] = df.mean()
	df.to_csv("exp_results.csv")
	
	#print(train_r2s)	
	print("train r squared mean: {}".format(df.iloc[-1, 1]))
	#print(val_r2s)		
	print("validation r squared mean: {}".format(df.iloc[-1, 2]))
	#print(aucs)
	print("AUC mean: {}".format(df.iloc[-1, 3]))
	

	