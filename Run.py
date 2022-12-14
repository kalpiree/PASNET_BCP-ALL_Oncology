from DataLoader import load_data, load_pathway
from Train import trainPASNet
from EvalFunc import auc, f1
import matplotlib.pyplot as plt

import torch
import numpy as np
from numpy import arange

dtype = torch.FloatTensor
Dropout_Rates = [0.7, 0.8]

''' Net Settings'''
In_Nodes = 1047  ###number of genes
# In_Nodes = 1508
Pathway_Nodes = 469  ###number of pathways
Hidden_Nodes = 100
Out_Nodes = 10  ###number of classes
useDropout = False

nEpochs = 15000  ###for training

K = 5  # number of folds

# Hyperparameters
# no pathway hypLr = [1e-3] hypL2 = [1e-6] hypHidden = [100]

# lr, l2, usePathway, shufflePathway
#hypCombis = [[1e-4, 1e-4, True, False], [1e-3, 1e-8, True, True], [1e-3, 1e-8, True, True], [1e-3, 1e-8, True, True]]
hypCombis = [1e-4, 1e-4, False, False]
names = ["pathway", "random #1", "random #2", "random #3"]
results = []
resultsMean = []
resultsStd = []
resultsTrain = []
resultsAucPlain = []
resultsAucMean = []
resultsAucStd = []

# for all hyperparamters
#for idx, model in enumerate(hypCombis):
modelResults = []
modelAuc = []
# Hyperparamters
opt_lr = hypCombis[0]
opt_l2 = hypCombis[1]
usePathwaydata = hypCombis[2]
shufflePathwaydata = hypCombis[3]
currHyper = "learning rate =  " + str(opt_lr) + ", l2 = " + str(opt_l2) + ", second hidden layer nodes = " + str(
        Hidden_Nodes)
print(currHyper)
''' load data and pathway '''
pathway_mask, npVersion = load_pathway("C:/Users/ntnbs/Downloads/Input_/Input/pathway.csv", dtype,
                                           shufflePathwaydata)
np.savetxt("C:/Users/ntnbs/Downloads/Input_/Input/current_results/pathway_mask.txt", npVersion, delimiter=",")

for fold in range(15):
    print("fold:", fold)
    x_train, y_train = load_data("C:/Users/ntnbs/Downloads/Input_/Input/fold" + str(fold + 3) + "_train.csv",
                                     dtype)  # x
        # "C:\Users\ntnbs\Downloads\Input_\Input\fold1_test.csv"
        # "C:\Users\ntnbs\Downloads\Input_\Input\fold1_train.csv"
        # "C:\Users\ntnbs\Downloads\hyper_vs_tcf\new_folds_hyper_vs_paxtrain_fold_1.csv"
        # "C:\Users\ntnbs\Downloads\new_folds\new_foldstrain_fold_1.csv"
        # "C:\Users\ntnbs\Downloads\Input_\Input\fold1_train.csv"
    x_test, y_test = load_data("C:/Users/ntnbs/Downloads/Input_/Input/fold" + str(fold + 3) + "_test.csv",
                                   dtype)  # x
        # "C:\Users\ntnbs\Downloads\new_foldstrain_fold_1.csv"

    pred_train, pred_test, loss_train, loss_test, f1_train, f1_test, f1train, f1eval = trainPASNet(usePathwaydata,
                                                                                                       useDropout,
                                                                                                       x_train, y_train,
                                                                                                       x_test, y_test,
                                                                                                       pathway_mask,
                                                                                                       In_Nodes,
                                                                                                       Pathway_Nodes,
                                                                                                       Hidden_Nodes,
                                                                                                       Out_Nodes,
                                                                                                       opt_lr, opt_l2,
                                                                                                       nEpochs,
                                                                                                       Dropout_Rates,
                                                                                                       optimizer="Adam")



        ###if gpu is being used, transferring back to cpu
    if torch.cuda.is_available():
        print("Using gpu!")
        pred_test = pred_test.cpu().detach()
        ###

    f1Score = (y_test, pred_test)
        # f1Score = np.asarray(f1Score)

    modelResults.append(f1Score)
    print(f1eval)
    print(type(modelResults))
    epochs = list(range(0,nEpochs,100))
    print(epochs)
    plt.plot(epochs,f1train, '-o')
    plt.plot(epochs,f1eval, '-o')
    plt.xlabel('epoch')
    plt.ylabel('F1 Score')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Accuracy')
    plt.xticks(arange(0, nEpochs, 1500))
    plt.show()
        # modelResults=np.asarray(modelResults)
        # modelResults.concatenate(f1Score)
    aucScore = auc(y_test, pred_test)
    modelAuc.append(aucScore)
        # modelAuc = np.asarray(modelAuc)
    # losses = [loss.detach().numpy() for loss in losses_all]
    # modelResults = [loss.detach().numpy() for loss in modelResults]
    # modelResults = modelResults.detach().numpy()
    # modelResults = np.array(modelResults)
    # modelResults = torch.Tensor(modelResults)
    # modelResults = [t.detach().numpy() for t in modelResults]
    # with torch.no_grad():
    # modelResults = [t.numpy() for t in modelResults]
modelResults = np.array(modelResults)
    # modelAuc=modelAuc.detach().np()
    # modelAuc = np.array(modelAuc)
    # modelAuc = torch.Tensor(modelAuc)
    # modelAuc = [t.detach().numpy() for t in modelAuc]
    # with torch.no_grad():
    # modelAuc = [x.numpy() for x in modelAuc]
modelAuc = np.array(modelAuc)
results.append(modelResults)
resultsMean.append(modelResults.mean())
resultsStd.append(modelResults.std())

resultsAucPlain.append(modelAuc)
resultsAucMean.append(modelAuc.mean())
resultsAucStd.append(modelAuc.std())

########################################################################################################################
# 06 save results in txt files
np.savetxt("C:/Users/ntnbs/Downloads/Input_/Input/current_results/resultsPlain.txt", results, delimiter=",")
# C:/Users/ntnbs/Downloads/Input_/Input/current_results/pathway_mask.txt
np.savetxt("C:/Users/ntnbs/Downloads/Input_/Input/current_results/resultsMean.txt", resultsMean, delimiter=",")
np.savetxt("C:/Users/ntnbs/Downloads/Input_/Input/current_results/resultsStd.txt", resultsStd, delimiter=",")
np.savetxt("C:/Users/ntnbs/Downloads/Input_/Input/current_results/resultsAucPlain.txt", resultsAucPlain, delimiter=",")
np.savetxt("C:/Users/ntnbs/Downloads/Input_/Input/current_results/resultsAucMean.txt", resultsAucMean, delimiter=",")
np.savetxt("C:/Users/ntnbs/Downloads/Input_/Input/current_results/resultsAucStd.txt", resultsAucStd, delimiter=",")

########################################################################################################################
# 07 boxplot algorithm comparison
fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names, rotation=30)
plt.xticks(fontsize=7)
plt.show()

# x = 1
