from evaluation import pearson_r_square, calculate_SSA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("vs/data/predictions.csv")
label = df.iloc[:, 0]
pred = df.iloc[:, 1]
#pred = np.mean(pred, 1)

r2 = pearson_r_square(pred, label)

print(r2)
#plot the correlation between predict and label of test

plt.xlabel('Predicted pIC50', fontsize=20)
plt.ylabel('True pIC50', fontsize=20)
plt.grid(True)		
plt.scatter(pred, label, c='b')
plt.text(3.0, 9, "$R^2$ = %4.2f"%r2, size=20, color="blue")	
plt.show()



	
	
#calculate sensitivity, selectivity and accuracy
roc_data = calculate_SSA(pred, label, 5, 2, 10, 0.2)
#print(roc_data)
#df = pd.DataFrame(roc_data, columns=["pIC50", "TPR", "FPR", "Accuracy"])
#df.to_csv("roc_table.csv")
auc = np.trapz(roc_data[::-1, 1], roc_data[::-1, 2])
print(auc)

#plot roc
	
plt.xlabel('FPR (1 - specificity)', fontsize=20)
plt.ylabel('TPR (sensitivity)', fontsize=20)
#plt.grid(True)	
plt.plot(roc_data[:, 2], roc_data[:, 1], 'g-')
plt.plot([0, 1], [0, 1], 'r-')
plt.text(0.6, 0.1, "AUC = %4.2f"%auc, size=20)
plt.show()
