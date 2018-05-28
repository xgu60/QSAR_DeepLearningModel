from predict import predict
import pandas as pd
import numpy as np 


def sceen(foldername, number_of_files):	
	collection = []
	for i in range(1, number_of_files):		
		pred = predict("vs/", "data/" + foldername + "/" + str(i) + "th.csv", [40, 45, 50])	
		pred = pred[np.where(pred[:, 1] > 4.5)]
		if pred.shape[0] > 0:
			#print(foldername)
			#print(i)
			print(pred)
			collection.append(pred)

	if len(collection) > 0:
		collection = np.vstack(collection)
	return collection

if __name__ == "__main__":
	filenames = ["smiles_0", "smiles_0_1", "smiles_1_2", "smiles_1_3","smiles_1_4",
		"smiles_1_5","smiles_1_6","smiles_1_7", "smiles_1_8", "smiles_1_9"]
	numbers = [135, 135, 130, 130, 131, 130, 130, 131, 130, 131]
	#numbers = [5] * 10
	res = []
	for i in range(len(filenames)):
		print(filenames[i])
		temp = sceen(filenames[i], numbers[i])
		if len(temp) > 0:
			res.append(temp)
	if len(res) > 0:
		res = np.vstack(res)
		df = pd.DataFrame(res, columns=["SMILE", "pIC50"])
		df.drop_duplicates(subset=['SMILE', 'pIC50'], keep='first', inplace=True)
		df.to_csv("vs/prediction/batch1.csv")
	else:
		print("no smiles found that match the search criteria!!!")
