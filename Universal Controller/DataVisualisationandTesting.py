import matplotlib
import matplotlib.pyplot as plt
import pickle
import datetime
from DataReporting import DataReporting
from MAPElites import MAPElites
from CMAES import CMAES
from deap import creator, base, tools, benchmarks, algorithms, cma
import numpy as np
from matplotlib.patches import Polygon
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import sem
from scipy.stats import t
from scipy.stats import ttest_ind
import seaborn as sns
import pandas as pd

'''Blank Python Module primarily for data loading and visualisation via the DataReporting class '''

'''BOXPLOT Exmaple from Sigma test data'''
#myData = DataReporting("Sigma Test - EscapeRoom - 150ME, 25PS, 25SR, 0.1-2.0, 07-21-2020, 12-17-57.txt")
myData = DataReporting("NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt")
'''Manually ensure the DataReporing class is set with relevant parameters ''' 
#myData.upperLimit = 2.0
#myData.lowerLimit = 0.1

myData.totalRuns = 10
myData.maxEvals = 10000
myData.yTicks = 1000
myData.loadData()

#myData.fullResults = [248, 2707, 5613, 5510, 13, 3354, 206, 802, -1, 1225]
#file = open("NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt",'wb')
#pickle.dump(myData.fullResults, file)
#file.close()

#testData = myData.displaySigmaBoxPlots()
testData = myData.displayAlgorithmBoxPlots("NCMA-ES")

#[-1, 2707, 5613, 5510, 13, 3354, 206, 802, -1, 1225]

'''BOXPLOT Exmaple from Algorithm Test Data'''
#myData = DataReporting("Algorithm Testing Data/CMA-ES/CMA-ES TEST - MiddleWall - 150ME, 25PS, 20TR, 07-21-2020, 00-28-46.txt")
#myData.totalRuns = 20
#myData.maxEvals = 150
#myData.loadData()
#myData.displayAlgorithmBoxPlots("CMA-ES", "NCMA-ES")
