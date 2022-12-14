import sklearn.metrics
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss


#def bce_for_one_class(predict, target, lts = False):
def bce_for_one_class(predict, target, idx):
	'''calculate cross entropy in average for samples who belong to the same class.
	lts = False: non-LTS samples are obtained.
	'''
	print("test")
	lts_idx = torch.argmax(target, dim = 1)
	#if lts == False:
		#idx = 0 # label = 0, non-LTS
	#else: idx = 1 # label = 1, LTS
	y = target[lts_idx == idx]
	pred = predict[lts_idx == idx]
	cost = F.binary_cross_entropy(pred, y)

	return(cost)

def binary_cross_entropy_for_imbalance(predict, target):
	'''claculate cross entropy for imbalance data in binary classification.'''
	# sum up binary for every class
	#total_cost = 0
	#for i in range(8): # cross entropy was binary, now multiclass x
		#total_cost += bce_for_one_class(predict, target, i)

	#return(total_cost)
	return(F.cross_entropy(predict,target))
