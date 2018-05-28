import torch
import numpy as np



def pearson_r_square(data_pred, data_label):
	label_mean = np.mean(data_label)
	pred_mean = np.mean(data_pred)
	numerator = np.sum((data_pred - data_label) ** 2)
	denominator = np.sum((data_label - label_mean) ** 2)
	return 1 - numerator / denominator


def calculate_SSA(pred, label, thres, low, high, step):
	#label and pred are numpy arrays	
	data = np.column_stack((label, pred))
	ROC_data = []
	for pic50 in np.arange(low, high, step):
		TP = data[np.where((data[:, 0] >= thres) * (data[:, 1] >= pic50))].shape[0]		
		FP = data[np.where((data[:, 0] < thres) * (data[:, 1] >= pic50))].shape[0]
		TN = data[np.where((data[:, 0] < thres) * (data[:, 1] < pic50))].shape[0]
		FN = data[np.where((data[:, 0] >= thres) * (data[:, 1] < pic50))].shape[0]		
		ROC_data.append([pic50, TP / (TP + FN), FP / (TN + FP), (TP + TN) / pred.shape[0]])
	return np.array(ROC_data)

def calculate_SSA_(pred, label, thres, low, high, step):
	#pred and label are numpy arrays	
	ROC_data = []
	for pic50 in np.arange(low, high, step):
		TP = 0.0
		FN = 0.0
		TN = 0.0
		FP = 0.0
		for i in range(pred.shape[0]):
			if label[i] >= thres and pred[i] >= pic50:
				TP += 1
			if label[i] < thres and pred[i] >= pic50:
				FP += 1
			if label[i] >= thres and pred[i] < pic50:
				FN += 1
			if label[i] < thres and pred[i] < pic50:
				TN += 1
		ROC_data.append([pic50, TP / (TP + FN), FP / (TN + FP), (TP + TN) / pred.shape[0]])
	return np.array(ROC_data)


if __name__ == "__main__":

	output = np.array([[3.0, 4.0], [4.0, 6.0], [6.0, 7.0]])
	print(calculate_SSA(output, 5, 3, 10, 0.5))

	'''
	data_pred = torch.Tensor([2.0, 3, 4])
	data_label = torch.Tensor([3, 5, 4])
	print (data_pred)
	print (data_label)
	
	print(pearson_r_square(data_pred, data_label))
	print(pearson_r(data_pred, data_label))
	'''

