from datalod import load_data, load_pathway
from Train import trainPASNet
from EvalFunc import auc, f1
import matplotlib.pyplot as plt

import torch
import numpy as np

dtype = torch.FloatTensor
Dropout_Rates = [0.7, 0.8]

''' Net Settings'''
In_Nodes = 4359  ###number of genes
Pathway_Nodes = 574  ###number of pathways
Hidden_Nodes = 100
Out_Nodes = 2  ###number of classes
useDropout = False

nEpochs = 80000  ###for training

K = 5  # number of folds

# Hyperparameters
# no pathway hypLr = [1e-3] hypL2 = [1e-6] hypHidden = [100]

# lr, l2, usePathway, shufflePathway
hypCombis = [[1e-3, 1e-6, True, False], [1e-3, 1e-8, True, True], [1e-3, 1e-8, True, True], [1e-3, 1e-8, True, True]]
names = ["pathway", "random #1", "random #2", "random #3"]
results = []
resultsMean = []
resultsStd = []
resultsTrain = []
resultsAucPlain = []
resultsAucMean = []
resultsAucStd = []

# for all hyperparamters
for idx, model in enumerate(hypCombis):
    modelResults = []
    modelAuc = []
    # Hyperparamters
    opt_lr = model[0]
    opt_l2 = model[1]
    usePathwaydata = model[2]
    shufflePathwaydata = model[3]
    currHyper = "learning rate =  " + str(opt_lr) + ", l2 = " + str(opt_l2) + ", second hidden layer nodes = " + str(
        Hidden_Nodes)
    print(currHyper)
    ''' load data and pathway '''
    pathway_mask, npVersion = load_pathway("C:/Users/ntnbs/Downloads/pathway_mask (1).csv", dtype,
                                           shufflePathwaydata)
    np.savetxt("C:/Users/ntnbs/Downloads/Input_/Input/current_results/pathway_mask.txt", npVersion, delimiter=",")

    #for fold in range(15):
        #print("fold: ", fold)
    x_train, y_train = load_data("C:/Users/ntnbs/Downloads/train (1).csv",
                                     dtype)  # x
    x_test, y_test = load_data("C:/Users/ntnbs/Downloads/validation (1).csv",
                                   dtype)  # x

    pred_train, pred_test, loss_train, loss_test, f1_train, f1_test = trainPASNet(usePathwaydata, useDropout,
                                                                                      x_train, y_train, x_test, y_test,
                                                                                      pathway_mask,
                                                                                      In_Nodes, Pathway_Nodes,
                                                                                      Hidden_Nodes, Out_Nodes,
                                                                                      opt_lr, opt_l2, nEpochs,
                                                                                      Dropout_Rates, optimizer="Adam")
        ###if gpu is being used, transferring back to cpu
    if torch.cuda.is_available():
        print("Using gpu!")
        pred_test = pred_test.cpu().detach()
        ###

    f1Score = f1(y_test, pred_test)
    modelResults.append(f1Score)

    aucScore = auc(y_test, pred_test)
    modelAuc.append(aucScore)

    modelResults = np.array(modelResults)
    modelAuc = np.array(modelAuc)

    results.append(modelResults)
    resultsMean.append(modelResults.mean())
    resultsStd.append(modelResults.std())

    resultsAucPlain.append(modelAuc)
    resultsAucMean.append(modelAuc.mean())
    resultsAucStd.append(modelAuc.std())

########################################################################################################################
# 06 save results in txt files
np.savetxt("current_results/resultsPlain.txt", results, delimiter=",")
np.savetxt("current_results/resultsMean.txt", resultsMean, delimiter=",")
np.savetxt("current_results/resultsStd.txt", resultsStd, delimiter=",")
np.savetxt("current_results/resultsAucPlain.txt", resultsAucPlain, delimiter=",")
np.savetxt("current_results/resultsAucMean.txt", resultsAucMean, delimiter=",")
np.savetxt("current_results/resultsAucStd.txt", resultsAucStd, delimiter=",")

########################################################################################################################
# 07 boxplot algorithm comparison
fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names, rotation=30)
plt.xticks(fontsize=7)
plt.show()

x = 1
