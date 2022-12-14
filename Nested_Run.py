from DataLoader import load_data, load_pathway
from Train import trainPASNet
from EvalFunc import auc, f1

import torch
import numpy as np

dtype = torch.FloatTensor
manuell = False

if not manuell:
	''' Net Settings'''
	In_Nodes = 1059 ###number of genes x
	Pathway_Nodes = 470 ###number of pathways x
	Out_Nodes = 9 ###number of hidden nodes in the last hidden layer x
	usePathwaydata = True
	nEpochs = 100000
	shufflePathwaydata = False

	''' Initialize '''
	Dropout_Rates = [0.8, 0.7] ###sub-network setup

	# Hyperparameters
	hypLr = [1e-5, 1e-4, 1e-3, 1e-2]
	hypL2 = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
	hypHidden = [25, 50, 100, 200]

	# all configurations to test
	hypCombis = []
	for lr in hypLr:
		for l2 in hypL2:
			for hidden in hypHidden:
				hypCombis.append([lr, l2, hidden])

	# for all test sets
	for outer in range(1,6):
		bestModel = None
		avgF1BestModel = float('-inf')
		avgF1BestTrainModel = float('-inf')
		avgF1BestValModel = float('-inf')

		x_test, y_test = load_data("../outer" + str(outer) + "_test.csv", dtype)

		useDropout = False

		# for hyperparameter combis
		for idx, model in enumerate(hypCombis):
			f1ScoresModel = []
			f1TrainModel = []
			f1ValModel = []

			# for all train/validation sets
			for inner in range(1,5):
				#print("Model " + str(idx + 1) + " with hyper " + str(model) + " Outer " + str(outer) + " Inner " + str(inner))
				x_train, y_train = load_data("../outer" + str(outer) + "_inner" + str(inner) + "_train.csv", dtype)
				x_val, y_val = load_data("../outer" + str(outer) + "_inner" + str(inner) + "_val.csv", dtype)

				# Hyperparamters
				opt_lr = model[0]
				opt_l2 = model[1]
				Hidden_Nodes = model[2]
				pathway_mask = load_pathway("../pathway.csv", dtype, shufflePathwaydata)

				pred_train, pred_val, loss_train, loss_val, f1_train, f1_val = trainPASNet(usePathwaydata, useDropout,
																		x_train, y_train, x_val, y_val, pathway_mask,
																		In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes,
																		opt_lr, opt_l2, nEpochs, Dropout_Rates,
																		optimizer="Adam")

				###if gpu is being used, transferring back to cpu
				if torch.cuda.is_available():
					pred_val = pred_val.cpu().detach()
				###

				f1Val = f1(y_val, pred_val)
				f1ScoresModel.append(f1Val)
				f1TrainModel.append(f1_val)
				f1ValModel.append(f1_train)

			# Average validation score of model
			avgF1ScoreModel = sum(f1ScoresModel) / len(f1ScoresModel)
			avgF1ValModel = sum(f1ValModel) / len(f1ValModel)
			avgF1TrainModel = sum(f1TrainModel) / len(f1TrainModel)
			# If average validation score better than current best model: update
			if avgF1ScoreModel > avgF1BestModel:
				bestModel = model
				avgF1BestModel = avgF1ScoreModel
				avgF1BestTrainModel = avgF1TrainModel
				avgF1BestValModel = avgF1ValModel

		# Know best model for validation score now
			# Really need to train again? Can't store model somehow?
		opt_lr = bestModel[0]
		opt_l2 = bestModel[1]
		nEpochs = bestModel[2]
		Hidden_Nodes = bestModel[3]
		usePathwaydata = bestModel[4]
		pathway_mask = load_pathway("../pathway.csv", dtype, shufflePathwaydata)
		pred_train, pred_test, loss_train, loss_test, f1_train, f1_test = trainPASNet(usePathwaydata, useDropout, x_train, y_train,
																 x_test, y_test, pathway_mask, \
																 In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, \
																 opt_lr, opt_l2, nEpochs, Dropout_Rates,
																 optimizer="Adam")
		###if gpu is being used, transferring back to cpu
		if torch.cuda.is_available():
			print("Using gpu!")
			pred_test = pred_test.cpu().detach()
		###

		f1Test = f1(y_test, pred_test)
		print("Model " + str(bestModel) + " achieved " + str(f1Test) + " in test" + "\n"
			  + "avg val " + str(avgF1BestValModel) + " avg train " + str(avgF1BestTrainModel))
