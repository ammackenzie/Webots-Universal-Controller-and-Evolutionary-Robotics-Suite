import matplotlib
import matplotlib.pyplot as plt
import pickle
import datetime
from DataReporting import DataReporting

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

'''Blank Python Module primarily for data loading and visualisation via the DataReporting class '''

'''BOXPLOT Exmaple from Sigma test data'''
#myData = DataReporting("Sigma Test - EscapeRoom - 150ME, 25PS, 25SR, 0.1-2.0, 07-21-2020, 12-17-57.txt")
myData = DataReporting("Sigma Testing Data/Sigma Test - MiddleWall - 150ME, 25PS, 25SR, 0.1-2.0, 07-21-2020, 00-28-46.txt")
'''Manually ensure the DataReporing class is set with relevant parameters ''' 
myData.upperLimit = 2.0
myData.lowerLimit = 0.1

myData.totalRuns = 25
myData.maxEvals = 150
myData.loadData()

testData = myData.displaySigmaBoxPlots()

'''BOXPLOT Exmaple from Algorithm Test Data'''
myData = DataReporting("Algorithm Testing Data/CMA-ES/CMA-ES TEST - MiddleWall - 150ME, 25PS, 20TR, 07-21-2020, 00-28-46.txt")
myData.totalRuns = 20
myData.maxEvals = 150
myData.loadData()
#myData.displayAlgorithmBoxPlots("CMA-ES", "CMA-ES")




# '''MANUAL SIGMA TEST CODE FOR REFERENCE'''
# creator.create("FitnessMin", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)
#  
# toolbox = base.Toolbox()
# toolbox.register("evaluate", benchmarks.rastrigin)
#  
# def main():
#     N = 200
#     np.random.seed()
#      
#      
#     strategy = cma.Strategy(centroid=[0]*N, sigma=1.0, lambda_= 50)
#     toolbox.register("generate", strategy.generate, creator.Individual)
#     toolbox.register("update", strategy.update)
#  
#     halloffame = tools.HallOfFame(1)
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("std", np.std)
#     stats.register("min", np.min)
#     stats.register("max", np.max)
#     
#     #algorithms.eaGenerateUpdate(toolbox, ngen=250, stats=stats, halloffame=hof)
#     count = []
#     temp = 0
#     under = 0
#     population = toolbox.generate()
#     for i in range(len(population)):
#         for j in range(len(population[i])):
#             if population[i][j] > 1.0:
#                 temp += 1
#             if population[i][j] < -1.0:
#                 under += 1
#                 temp += 1
#         count.append(temp)
#         temp = 0
#     print(np.array(count))
#     print(under / len(population))
#     # Evaluate the individuals
#     fitnesses = toolbox.map(toolbox.evaluate, population)
#     for ind, fit in zip(population, fitnesses):
#         ind.fitness.values = fit
#      
#     halloffame.update(population)
#     record = stats.compile(population)
#      
#     # Update the strategy with the evaluated individuals
#     toolbox.update(population)
#  
# main()