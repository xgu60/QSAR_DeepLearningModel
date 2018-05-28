from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np
import pandas as pd
import matplotlib

def featurizer(folder, filename):
	folderpath = folder
	data = pd.read_csv(folderpath + filename, sep=',')

	alias = []
	ic50 = []
	fps = []

	#convert smiles to rkd objects
	for i in range(data.shape[0]):	
		try:
			arr = np.zeros((1, ))
			compound = Chem.MolFromSmiles(data.ix[i, 1])
			fp = AllChem.GetMorganFingerprintAsBitVect(compound, 3, 4096)
			DataStructs.ConvertToNumpyArray(fp, arr)
			fps.append(arr)
			alias.append(data.ix[i, 0])
			ic50.append(data.ix[i, 2])
		except:
			print(i)
			print(data.ix[i, 0])
	
	#create dataframe to store fingerprinters and write to csv file
	df = pd.DataFrame(fps, index=alias)
	df.insert(0, "ic50", ic50)
	df.to_csv(folderpath + 'fingerprinters_4096.csv')

if __name__ == "__main__":
	featurizer("ccf/data/test/", "test_data.csv")


