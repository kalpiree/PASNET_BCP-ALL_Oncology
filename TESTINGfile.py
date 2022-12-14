from DataLoader import load_data, load_pathway
import pandas as pd
import numpy as np
from Train import trainPASNet
from EvalFunc import auc, f1
import matplotlib.pyplot as plt

import torch
import numpy as np
x = pd.read_csv("C:/Users/ntnbs/Downloads/Input_/Input/pathway.csv")