from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np
import pandas as pd
import matplotlib

		

def draw(data, savefolder):
	#data = pd.read_csv(filename)
	mols = []	
	for i in range(data.shape[0]):	
		try:
			compound = Chem.MolFromSmiles(data.iloc[i, 1])
			mols.append(compound)		
		except:
			print(i)
			print(data.iloc[i, 1])
	
	if len(mols) > 40:
		for i in range(1, int(len(mols) / 40) + 1):
			img = Draw.MolsToGridImage(mols[(i - 1) * 40 : i * 40], molsPerRow=4, subImgSize=(400, 400))
			img.save(savefolder + "/" + str(i) + ".png")
		img = Draw.MolsToGridImage(mols[i * 40 : ], molsPerRow=4, subImgSize=(400, 400))
		img.save(savefolder + "/" + str(i + 1) + ".png")

	else:
		img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(400, 400))
		img.save(savefolder + "/0.png")

if __name__ == "__main__":
	ori_data = pd.read_csv("vs/prediction/batch1.csv")
	df1 = ori_data[(ori_data['pIC50'] >= 6)]
	df2 = ori_data[(ori_data['pIC50'] >= 5) & (ori_data['pIC50'] < 6)]
	df3 = ori_data[(ori_data['pIC50'] >=4.5) & (ori_data['pIC50'] < 5)]
	print(df1.shape)
	print(df2.shape)
	print(df3.shape)
	draw(df1, "vs/prediction/batch1_6")
	draw(df2, "vs/prediction/batch1_5")
	draw(df3, "vs/prediction/batch1_4.5")


