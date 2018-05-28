from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np
import pandas as pd
import matplotlib

def batch_featurizer(folderpath, inputfilename, outputfolderpath):	
	ori_data = pd.read_csv(folderpath + inputfilename, sep=',')
	batch_num = int(ori_data.shape[0] / 1000)
	print(batch_num)
	for batch in range(1, batch_num + 1):
		data = ori_data[(batch - 1) * 1000 : batch * 1000]
		featurizer(data, folderpath + outputfolderpath + str(batch) + "th.csv")
	data = ori_data[1000 * batch :]
	featurizer(data, folderpath + outputfolderpath + str(batch + 1) + "th.csv")
		

def featurizer(data, outputfilename):
	alias = []
	smiles = []	
	fps = []
	#convert smiles to rkd objects
	for i in range(data.shape[0]):	
		try:
			arr = np.zeros((1, ))
			compound = Chem.MolFromSmiles(data.iloc[i, 1])
			fp = AllChem.GetMorganFingerprintAsBitVect(compound, 3, 4096)
			DataStructs.ConvertToNumpyArray(fp, arr)
			fps.append(arr)
			alias.append(data.iloc[i, 0])
			smiles.append(data.iloc[i, 1])			
		except:
			print(i)
			print(data.ix[i, 0])
	
	#create dataframe to store fingerprinters and write to csv file
	df = pd.DataFrame(fps, index=alias)
	df.insert(0, "smiles", smiles)	
	df.to_csv(outputfilename)

if __name__ == "__main__":
	for i in range(2, 10):
		print(i)
		batch_featurizer("vs/data/", "smiles_1_" + str(i) + ".csv", "smiles_1_" + str(i) + "/")


