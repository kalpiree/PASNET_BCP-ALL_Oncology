import numpy as np
import pandas as pd
import torch



def vectorized_label(target, n_class):
	'''convert target(y) to be one-hot encoding format(dummy variable)
	'''
	TARGET = np.array(target).reshape(-1)

	#dict = {'G2_TCF3-PBX1': 0, 'G3_ETV6-RUNX1': 1, 'G9_PAX5': 2, 'G9_CRLF2-P2RY8': 3, 'G8_MLL': 4,
	#		'G6_BCR-ABL': 5, 'G5_ZNF384': 6, 'G7_hyper>49': 7} # x
	dict = {'TCF3-PBX1': 0, 'ETV6-RUNX1': 1, 'PAX5': 2, 'CRLF2-P2RY8': 3, 'KMT2A': 4, 'BCR-ABL': 5, 'ZNF384': 6, 'MEF2D': 7,
			'hyper>49 or 50': 8, 'DUX4': 9}
	#dict = {'hyper>49 or 50': 0, 'TCF3-PBX1': 1}
	TARGET = [dict[i] for i in TARGET] # x

	return np.eye(n_class)[TARGET]


def load_data(path, dtype):
	'''Load data, and then covert it to a Pytorch tensor.
	Input:
		path: path to input dataset (which is expected to be a csv file).
		dtype: define the data type of tensor (i.e. dtype=torch.FloatTensor)
	Output:
		X: a Pytorch tensor of 'x'.
		Y: a Pytorch tensor of 'y'(one-hot encoding).
	'''
	data = pd.read_csv(path)

	x = data.drop(["Subgroup Name"], axis=1)  # x
	x.drop(columns=x.columns[0], inplace=True)  # x don't know why extra first column
	x = x.values  # x
	y = data.loc[:, ["Subgroup Name"]].values  # x
	X = torch.from_numpy(x).type(dtype)
	Y = torch.from_numpy(vectorized_label(y, 10)).type(dtype) # Changed if number classes changes
	###if gpu is being used
	if torch.cuda.is_available():
		X = X.cuda()
		Y = Y.cuda()
	###
	return(X, Y)


def load_pathway(path, dtype, shufflePathwaydata):
	'''Load a bi-adjacency matrix of pathways, and then covert it to a Pytorch tensor.
	Input:
		path: path to input dataset (which is expected to be a csv file).
		dtype: define the data type of tensor (i.e. dtype=torch.FloatTensor)
	Output:
		PATHWAY_MASK: a Pytorch tensor of the bi-adjacency matrix of pathways.
	'''
	pathway_mask = pd.read_csv(path, index_col = 0).values

	# Does shuffle affect ?
	if shufflePathwaydata:
		for row in pathway_mask:
			np.random.shuffle(row)
		for column in pathway_mask.T:
			np.random.shuffle(column)

	PATHWAY_MASK = torch.from_numpy(pathway_mask).type(dtype)
	###if gpu is being used
	if torch.cuda.is_available():
		PATHWAY_MASK = PATHWAY_MASK.cuda()
	###
	return(PATHWAY_MASK, pathway_mask)
