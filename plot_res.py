import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("exp_results.csv")

plt.subplot(121)
plt.xlabel('experiments', fontsize=20)
plt.ylabel('results', fontsize=20)
plt.ylim((0,1))
#plt.grid(True)	
plt.scatter(range(0,10), df.train_r2.values[:-1], c='r', label="training $R^2$", alpha=0.5)
plt.scatter(range(0,10), df.val_r2.values[:-1], c='b', label="training $R^2$")
#plt.scatter(range(0,100), df.auc.values[:-1], c='gold', label="AUC")
plt.text(1, 0.7, "training $R^2$ mean = %4.2f"%df.iloc[10][2], size=16, color="r")
#plt.text(1, 0.7, "AUC mean = %4.2f"%df.iloc[100][4], size=16, color="b")
plt.text(1, 0.6, "validation $R^2$ mean = %4.2f"%df.iloc[10][3], size=16, color="b")

plt.subplot(122)
plt.xlabel('experiments', fontsize=20)
plt.ylabel('results', fontsize=20)
plt.ylim((0,1))
#plt.grid(True)	
#plt.scatter(range(0,100), df.train_r2.values[:-1], c='r', label="training $R^2$", alpha=0.5)
#plt.scatter(range(0,100), df.val_r2.values[:-1], c='b', label="training $R^2$")
plt.scatter(range(0,10), df.auc.values[:-1], c='g', label="AUC")
#plt.text(1, 0.95, "training $R^2$ mean = %4.2f"%df.iloc[100][2], size=16, color="r")
plt.text(1, 0.5, "AUC mean = %4.2f"%df.iloc[10][4], size=16, color="g")
#plt.text(1, 0.5, "validation $R^2$ mean = %4.2f"%df.iloc[100][3], size=16, color="r")
plt.tight_layout()
plt.show()