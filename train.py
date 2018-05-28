from data_loader import QSAR_Dataset, getValidationData, getTestData
from preprocess import shuffle_dataframe
from net_structure import DNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd 
import matplotlib.pyplot as plt

def train_model(train, validation, visual=False, epoch=51, batchSize=32, saveNet=True,
					savePath="ccf/net/", savePoints=[40, 45, 50]):
	EPOCH_NUM = epoch
	BATCH_SIZE = batchSize
	FEATURES = 4096
	OUTPUTS = 1	
	NET_SAVE_POINTS = savePoints


	train_dataset = QSAR_Dataset(train)	
	train_data = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
		num_workers=2)
	val_data, val_label = getValidationData(validation)
	

	net = DNN(FEATURES, OUTPUTS).cuda()	
	val_data = Variable(val_data.cuda())
	val_label = Variable(val_label.cuda())

	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr=0.05, betas=(0.9, 0.99))

	if visual:
		epochs = []
		train = []
		validation = []
		plt.ion()
		plt.xlabel('epoch', fontsize=16)
		plt.ylabel('mse loss', fontsize=16)
		plt.yscale('log')
		plt.grid(True)

	for epoch in range(EPOCH_NUM):
		for i, (inputs, labels) in enumerate(train_data):
			net.train()
			inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
			y_pred = net(inputs)
			loss = criterion(y_pred, labels)					

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()	

			if saveNet and epoch in NET_SAVE_POINTS and i == 0:
				torch.save(net, savePath + 'epoch' + str(epoch) + '.pkl')

			if epoch % 1 == 0 and i == 0:
				#evaluate the trained network
				net.eval()
				y_pred = net(inputs)
				train_loss = criterion(y_pred, labels).data

				val_pred = net(val_data)
				val_loss = criterion(val_pred, val_label).data
				print((train_loss[0], val_loss[0]))
			
				if visual:
					#update plot					
					epochs.append(epoch)
					train.append(train_loss[0])
					validation.append(val_loss[0])		
					plt.plot(epochs, train, 'b-')
					plt.plot(epochs, validation, 'r-')
					plt.pause(0.01)

	if visual:
		plt.plot(epochs, train, 'b-', label='training')
		plt.plot(epochs, validation, 'r-', label='validation')
		plt.legend(loc='upper right', shadow=True)
		plt.ioff()
		
		plt.show()

if __name__ == "__main__":
	df = pd.read_csv("ccf/data/processed_data.csv")
	train, val = shuffle_dataframe(df, True, 6, 0.7)
	train_model(train, val, visual=True, epoch=20)

	
	




	 
		    

    

    





