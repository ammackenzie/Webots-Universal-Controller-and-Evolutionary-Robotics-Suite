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









# myData = DataReporting("Algorithm Testing Data/SEND-THIS-TO-DREW-H-CMAES-FNN-Easy-Race-0-10000ME-50PS-1TR-08-03-2020-17-47-53.txt")
# myData.loadData()
# print(myData.data)
''' HEATMAP FOR EASY RACE E PUCK FNN'''

# def myfunc():
#     pass
#  
# myMap = MAPElites(58, myfunc, binCount = 10, tournamentSize = 5, maxVelocity = 0.125, maxDistance = 1.2)
#  
# myMap.loadMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 9 5TS, 10BC, 10000gens08-05-2020, 02-47-57.txt")
#  
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 8 5TS, 10BC, 10000gens08-05-2020, 01-55-06.txt")
#  
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 7 5TS, 10BC, 10000gens08-05-2020, 00-05-48.txt")
#  
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 6 5TS, 10BC, 10000gens08-04-2020, 23-13-05.txt")
#  
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 5 5TS, 10BC, 10000gens08-04-2020, 21-48-43.txt")
#  
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 4 5TS, 10BC, 10000gens08-04-2020, 20-13-12.txt")
#  
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 3 5TS, 10BC, 10000gens08-04-2020, 20-04-07.txt")
#  
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 2 5TS, 10BC, 10000gens08-04-2020, 19-17-57.txt")
#  
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 1 5TS, 10BC, 10000gens08-04-2020, 18-09-00.txt")
#  
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 0 5TS, 10BC, 10000gens08-04-2020, 17-45-56.txt")
#  
# print(myMap.searchSpace)
# df = pd.DataFrame(myMap.searchSpace)
# ax = sns.heatmap(df)
# ax.set_xticklabels(myMap.distanceFromStart,
#                             rotation=0, fontsize=12)
# ax.set_yticklabels(myMap.velocity,rotation=0, fontsize=12)
# ax.set_xlabel("Distance from Start (m)", fontsize=14)
# ax.set_ylabel("Average Velocity (m/s)", fontsize=14)
# ax.set_title("MAP-Elites Heatmap - Easy Race - Custom e-puck - FNN", fontsize=14)
# #sns.heatmap(df, xticklabels=1)
# plt.show()

''' MAP-ELITES EASY RACE BOX PLOT'''

# myData = DataReporting()
#  
# myData.data = [[2175, 448, 1321, 818, 169, 1909, 1726, 1080, 2224, 1174], [1059, 1280, 3572, 1503, 494, 1138, 1474, 555, 1229, 141], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
# myData.totalRuns = 10
# myData.maxEvals = 10000
# myData.yTicks = 1000
# myData.graphPadding = 1000
# myData.xAxisLabel = "Agent/Neural Network"
# myData.yAxisLabel = "Number of Samples"
#  
#  
# testData = myData.displayAlgorithmBoxPlots("MAP-Elites - Easy Race", "e-puck FNN", "e-puck RNN", "Hemisson FNN", "Hemisson RNN")
# 
# 
#  
''' HEATMAP FOR EASY RACE E PUCK RNN'''
 
# def myfunc():
#     pass
# 
# myMap = MAPElites(122, myfunc, binCount = 10, tournamentSize = 5, maxVelocity = 0.125, maxDistance = 1.2)
# 
# myMap.loadMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 9 5TS, 10BC, 10000gens08-05-2020, 02-33-59.txt")
# 
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 8 5TS, 10BC, 10000gens08-05-2020, 02-27-00.txt")
# 
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 7 5TS, 10BC, 10000gens08-05-2020, 01-25-57.txt")
# 
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 6 5TS, 10BC, 10000gens08-05-2020, 00-58-13.txt")
# 
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 5 5TS, 10BC, 10000gens08-04-2020, 23-45-28.txt")
# 
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 4 5TS, 10BC, 10000gens08-04-2020, 22-49-12.txt")
# 
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 3 5TS, 10BC, 10000gens08-04-2020, 22-24-43.txt")
# 
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 2 5TS, 10BC, 10000gens08-04-2020, 21-08-37.txt")
# 
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 1 5TS, 10BC, 10000gens08-04-2020, 17-57-06.txt")
# 
# myMap.combineLoadedMap("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 0 5TS, 10BC, 10000gens08-04-2020, 16-48-35.txt")
# 
# 
# myMap.heatmapTitle = "MAP-Elites Heatmap - Easy Race - Custom e-puck - RNN" 
# # print(myMap.searchSpace)
# # df = pd.DataFrame(myMap.searchSpace)
# # ax = sns.heatmap(df)
# # ax.set_xticklabels(myMap.distanceFromStart,
# #                           rotation=0, fontsize=12)
# # ax.set_yticklabels(myMap.velocity,rotation=0, fontsize=12)
# # ax.set_xlabel("Distance from Start (m)", fontsize=14)
# # ax.set_ylabel("Average Velocity (m/s)", fontsize=14)
# # ax.set_title("MAP-Elites Heatmap - Easy Race - Custom e-puck - RNN", fontsize=14)
# # #sns.heatmap(df, xticklabels=1)
# # plt.show()
# 
# myMap.generateHeatmap()



'''Blank Python Module primarily for data loading and visualisation via the DataReporting class '''


# test = np.random.uniform(-1, 1, 100)
# print(test)

'''BOXPLOT Exmaple from Sigma test data'''

# def mefunc():
#     pass
#  
# #tempEA = MAPElitesMVDFS(122, mefunc)
# 
# count = 0
# for j in range(20):
#     for i in range(20):
#         if i <= j:
#             pass
#         else:
#             count += 1
# 
# print(count)
# 
# for i in range(20):
#     print(i)
# myData = DataReporting("MAPElites easy race FNN source1 0 5TS, 10BC, 10gens 08-03-2020, 21-13-36.txt")
# myData.loadData()
# print(myData.data)

#12 + 24 +  

''' Sigma significance testing line graph code'''

#Escape room: ({1.4: 7, 1.3: 4, 0.6: 3, 1.0: 2, 1.9: 2, 0.2: 1, 0.5: 1, 0.8: 1, 0.9: 1, 1.2: 1, 1.7: 1})
# Middle Wall: ({1.7: 7, 1.3: 5, 1.9: 5, 0.5: 5, 1.8: 5, 2.0: 5, 0.3: 4, 1.0: 4, 1.2: 3, 1.6: 3, 0.4: 1, 0.6: 1})
# Multi Maze: ({2.0: 6, 0.3: 2, 1.7: 1, 0.6: 1, 0.7: 1, 1.3: 1, 1.4: 1, 1.8: 1})
# 
#  myEA = CMAES(200, mefunc)
#  
#  print(myEA.population)
#  myData = DataReporting()
#  #myData.loadData()
#  myData.upperLimit = 2.0
#  myData.lowerLimit = 0.1
#  myData.maxEvals = 65
#  myData.yTicks = 5
#  myData.xAxisLabel = "Sigma Value"
#  myData.yAxisLabel = "Total Null Hypothesis Rejections"
# myData.data = [0.1:0, 0.2:1, 0.3:6, 0.4:1, 0.5:6, 0.6:5, 0.7:1, 0.8:1, 0.9:1, 1.0:4, 1.1:0, 1.2:4, 1.3:10, 1.4:8, 1.5:0, 1.6:3, 1.7:9, 1.8:1, 1.9:5, 2.0:11]
# myData.data = [0, 1, 6, 1, 6, 5, 1, 1, 1, 4, 0, 4, 10, 8, 0, 3, 9, 1, 5, 11]
# 
# 
# myData.displayLineGraph("Sigma Significance Testing (p value and Student's T Testing)")

''' sigma clipping graph'''
# 
 
 
# myData = DataReporting()
# #myData.loadData()
# myData.upperLimit = 2.0
# myData.lowerLimit = 0.1
# myData.maxEvals = 100
# myData.yTicks = 10
# myData.xAxisLabel = "Sigma Value"
# myData.yAxisLabel = "Percentage of Avg 'Clipped' Genes"
# myData.data = [0.0, 0.0, 0.09, 1.04, 4.69, 9.7, 15.1, 21.31, 26.8, 30.85, 36.49, 40.25, 45.33, 48.2, 50.57, 52.36, 55.54, 57.8, 59.62, 61.46]
#  
#  
# myData.displayLineGraph("Proportion of 'Clipped' Genes per Individual in Initial CMA-ES Pop")


# #tempEA.saveFile = "EasyRace initial test DFS 1K TS5.txt"
# #tempEA.loadMap("EasyRace initial test DFS 1K TS5.txt")
# 
# #solutions = tempEA.visualiseMap()
# 
# #print(solutions)
# 
# successes = []
# 
# #newMember = solutions[0]
# coordinates = [3, 3]
# gen = 100
# 
# 
# robot = np.array([0.000140432, -4.53794e-05])
# 
# farpoint = np.array([0.443902, -0.443007])
# 
# farpoint2 = np.array([])
# maxd = np.linalg.norm(robot - farpoint) 
# 
# print(maxd)

#print(tempEA.velocity)
#print(tempEA.distanceFromStart)
#successes.append([newMember, coordinates, gen])

#print(successes[0])
#print(len(successes))
#myData = DataReporting("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-24-2020, 09-21-51.txt")
#myData = DataReporting("Sigma Testing Data/Sigma Test - MiddleWall - 150ME, 25PS, 25SR, 0.1-2.0, 07-21-2020, 00-28-46.txt")
#myData = DataReporting("Sigma Testing Data/Sigma Test - MultiMaze - 300ME, 50PS, 25SR, 0.1-2.0, 07-23-2020, 03-46-21.txt")

#myData = DataReporting("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, FNN, - 10000ME, 50PS, 10TR,  07-29-2020, 10-12-18.txt")
#myData = DataReporting("Algorithm Testing Data/NCMA-ES/NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt")
'''Manually ensure the DataReporing class is set with relevant parameters ''' 
#myData.upperLimit = 2.0
#myData.lowerLimit = 0.1
# 
# myData.totalRuns = 10
# myData.maxEvals = 550
# myData.yTicks = 100
# myData.graphPadding = 10
# myData.xAxisLabel = "Agent/Neural Network"
# myData.yAxisLabel = "Number of Samples"
# myData.loadData()
# # 
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 10-29-26.txt")
# 
# 
# myData.loadData("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-29-2020, 20-09-36.txt")
# 
# myData.loadData("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 01-55-53.txt")

#myData.loadData("Algorithm Testing Data/CMA-ES/H-CMAES FULL TEST - EscapeRoom, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 20-17-18.txt")

#myData.fullResults = [248, 2707, 5613, 5510, 13, 3354, 206, 802, -1, 1225]
#file = open("NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt",'wb')
#pickle.dump(myData.fullResults, file)
#file.close()





#testData = myData.displaySigmaBoxPlots("Sigma Testing - Escape Room")
#testData = myData.displayAlgorithmBoxPlots("N-CMA-ES TEST - Multi Maze", "e-puck FNN", "e-puck RNN", "Hemisson FNN", "Hemisson RNN")

#[-1, 2707, 5613, 5510, 13, 3354, 206, 802, -1, 1225]



'''SIGMA CODE '''
# myData = DataReporting("Sigma Testing Data/Sigma Test - EscapeRoom - 150ME, 25PS, 25SR, 0.1-2.0, 07-21-2020, 12-17-57.txt")
# #myData = DataReporting("Sigma Testing Data/Sigma Test - MiddleWall - 150ME, 25PS, 25SR, 0.1-2.0, 07-21-2020, 00-28-46.txt")
# #myData = DataReporting("Sigma Testing Data/Sigma Test - MultiMaze - 300ME, 50PS, 25SR, 0.1-2.0, 07-23-2020, 03-46-21.txt")
#    
# myData.upperLimit = 2.0
# myData.lowerLimit = 0.1
#  
# myData.totalRuns = 25
# myData.maxEvals = 300
# myData.yTicks = 100
# myData.xAxisLabel = "Sigma Value"
# myData.yAxisLabel = "Number of Samples"
# myData.graphPadding = 20
# myData.loadData()
# testData = myData.displaySigmaBoxPlots("Sigma Testing - Middle Wall")
'''BOXPLOT Exmaple from Algorithm Test Data'''
#myData = DataReporting("Algorithm Testing Data/CMA-ES/CMA-ES TEST - MiddleWall - 150ME, 25PS, 20TR, 07-21-2020, 00-28-46.txt")
#myData.totalRuns = 20
#myData.maxEvals = 150
#myData.loadData()
#myData.displayAlgorithmBoxPlots("CMA-ES", "NCMA-ES")




'''MANUAL SIGMA TEST CODE'''
# creator.create("FitnessMin", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)
#   
# toolbox = base.Toolbox()
# toolbox.register("evaluate", benchmarks.rastrigin)
#   
# def main():
#     N = 200
#     np.random.seed()
#     proportions = []
#     
#     for sig in (round(i * 0.1, 1) for i in range(round(0.1*10), round(2.0*10)+1)):
#         strategy = cma.Strategy(centroid=[0]*N, sigma=sig, lambda_= 50)
#         toolbox.register("generate", strategy.generate, creator.Individual)
#         toolbox.register("update", strategy.update)
#       
#         halloffame = tools.HallOfFame(1)
#         stats = tools.Statistics(lambda ind: ind.fitness.values)
#         stats.register("avg", np.mean)
#         stats.register("std", np.std)
#         stats.register("min", np.min)
#         stats.register("max", np.max)
#          
#         #algorithms.eaGenerateUpdate(toolbox, ngen=250, stats=stats, halloffame=hof)
#         count = []
#         temp = 0
#         clipped = 0
#         population = toolbox.generate()
#         for i in range(len(population)):
#             for j in range(len(population[i])):
#                 if population[i][j] > 1.0:
#                     temp += 1
#                 if population[i][j] < -1.0:
#                     clipped += 1
#                     temp += 1
#             count.append(temp)
#             temp = 0
#         #print(np.array(count))
#         avgclipped = np.sum(np.array(count)) / len(count)
#         #print(avgclipped)
#         percentage = round((avgclipped/N) * 100, 2)
#         print(percentage)
#         proportions.append(percentage)
#     
#     print(proportions)
    # Evaluate the individuals
#     fitnesses = toolbox.map(toolbox.evaluate, population)
#     for ind, fit in zip(population, fitnesses):
#         ind.fitness.values = fit
#       
#     halloffame.update(population)
#     record = stats.compile(population)
#       
#     # Update the strategy with the evaluated individuals
#     toolbox.update(population)
  
#main()



''' NCMAES MULTI MAZE CODE '''
#myData = DataReporting("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, FNN, - 10000ME, 50PS, 10TR,  07-29-2020, 10-12-18.txt")

'''Manually ensure the DataReporing class is set with relevant parameters ''' 

# myData.totalRuns = 10
# myData.maxEvals = 550
# myData.yTicks = 100
# myData.graphPadding = 10
# myData.xAxisLabel = "Agent/Neural Network"
# myData.yAxisLabel = "Number of Samples"
# myData.loadData()
# # 
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 10-29-26.txt")
# 
# 
# myData.loadData("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-29-2020, 20-09-36.txt")
# 
# myData.loadData("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 01-55-53.txt")
# 
# 
# testData = myData.displayAlgorithmBoxPlots("N-CMA-ES TEST - Multi Maze", "e-puck FNN", "e-puck RNN", "Hemisson FNN", "Hemisson RNN")
# 
''' NCMAES EASY RACE CODE '''
 
# myData = DataReporting("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-24-2020, 09-21-51.txt")
#  
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#  
# myData.totalRuns = 10
# myData.maxEvals = 10000
# myData.yTicks = 1000
# myData.graphPadding = 1000
# myData.xAxisLabel = "Agent/Neural Network"
# myData.yAxisLabel = "Number of Samples"
# myData.loadData()
# # 
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt")
# 
# myData.loadData("Algorithm Testing Data/NCMA-ES/FOR-TOM-SEND-2-DREW-1-HEMISSON-EASY-RACE-RNN-10000ME-50PS-1TR-07-30-2020-11-08-54.txt")
# myData.loadData("Algorithm Testing Data/NCMA-ES/FOR-TOM-SEND-2-DREW-1-HEMISSON-EASY-RACE-RNN-10000ME-50PS-1TR-07-30-2020-11-08-54.txt")
# 
# testData = myData.displayAlgorithmBoxPlots("N-CMA-ES TEST - Easy Race", "e-puck FNN", "e-puck RNN", "Hemisson FNN", "Hemisson RNN")



''' NCMAES Escape Room CODE '''
# myData = DataReporting("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EscapeRoom, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-29-2020, 12-37-46.txt")
# 
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
# 
# myData.totalRuns = 10
# myData.maxEvals = 30
# myData.yTicks = 5
# myData.graphPadding = 10
# myData.xAxisLabel = "Agent/Neural Network"
# myData.yAxisLabel = "Number of Samples"
# myData.loadData()
# # 
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EscapeRoom, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-29-2020, 12-45-20.txt")
#  
#  
# myData.loadData("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - EscapeRoom, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 19-36-28.txt")
#  
# myData.loadData("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - EscapeRoom, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 20-00-33.txt")
#  
#  
# testData = myData.displayAlgorithmBoxPlots("N-CMA-ES TEST - Escape Room", "e-puck FNN", "e-puck RNN", "Hemisson FNN", "Hemisson RNN")


''' COMBINED EASY RACE CODE '''
    
# myData = DataReporting("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-25-2020, 22-38-31.txt")
#            
# myData.totalRuns = 10
# myData.maxEvals = 10000
# myData.yTicks = 1000
# myData.graphPadding = 5000
# myData.xAxisLabel = "Algorithm/Neural Network"
# myData.yAxisLabel = "Number of Samples"
# myData.blueLabel = "Fixed NN"
# myData.tanLabel = "Recurrent NN"
# myData.loadData()
#     
# myData.loadData("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 01-49-49.txt")
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-24-2020, 09-21-51.txt")
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt")
#     
# #map elites
# myData.data.append([2175, 448, 1321, 818, 169, 1909, 1726, 1080, 2224, 1174])
#       
# myData.data.append([1059, 1280, 3572, 1503, 494, 1138, 1474, 555, 1229, 141])
#      
# #control group
# myData.data.append([4650, 10000, 2366, 5472, 5469, 2738, 7863, 5422, 10000, 1769])
#     
# myData.data.append([232, 1036, 4117, 182, 4314, 3944, 7574, 3577, 3865, 2807])
#   
# for i in range(len(myData.data)):
#     for j in range(len(myData.data)):
#         if j <= i or j > (i+ 1):
#             pass
#         else:
#             myData.basicTtest(myData.data[i], myData.data[j]) 
#             print("compared " + str(i) + " to " + str(j))
#        
# 
#  
# 
#      
# testData = myData.displayAlgorithmBoxPlots(["CMA-ES", " CMA-ES", "N-CMA-ES", "N-CMA-ES", "MAP-Elites", "MAP-Elites", "Control", "Control"], "Algorithms Comparison - Easy Race - e-puck only")
#  


''' COMBINED EASY RACE RNN CODE with control group '''
# #  
# myData = DataReporting("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 01-49-49.txt")
#       
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#       
# myData.totalRuns = 10
# myData.maxEvals = 10000
# myData.yTicks = 1000
# myData.graphPadding = 5000
# myData.xAxisLabel = "Algorithm"
# myData.yAxisLabel = "Number of Samples"
# myData.loadData()
# # 
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt")
#     
#     
# myData.data.append([1059, 1280, 3572, 1503, 494, 1138, 1474, 555, 1229, 141])
#    
# #control group RNN
# myData.data.append([232, 1036, 4117, 182, 4314, 3944, 7574, 3577, 3865, 2807])
#    
#    
# testData = myData.displayAlgorithmBoxPlots(["CMA-ES", "N-CMA-ES", "MAP-Elites", "Control"], "Recurrent Neural Network Results - Easy Race")



''' COMBINED EASY RACE FNN CODE'''
#   
# myData = DataReporting("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-25-2020, 22-38-31.txt")
#       
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#       
# myData.totalRuns = 10
# myData.maxEvals = 10000
# myData.yTicks = 1000
# myData.graphPadding = 5000
# myData.xAxisLabel = "Algorithm"
# myData.yAxisLabel = "Number of Samples"
# myData.loadData()
# # 
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-24-2020, 09-21-51.txt")
#  
# #MAP ELITES
# myData.data.append([2175, 448, 1321, 818, 169, 1909, 1726, 1080, 2224, 1174])
#     
# #CONTROL  
# myData.data.append([4650, 10000, 2366, 5472, 5469, 2738, 7863, 5422, 10000, 1769])
#  
# testData = myData.displayAlgorithmBoxPlots(["CMA-ES", "N-CMA-ES", "MAP-Elites", "Control"], "Fixed Neural Network Results - Easy Race")
#    
# 



''' COMBINED EASY RACE FULL NN CODE'''
   
# myData = DataReporting("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-25-2020, 22-38-31.txt")
#        
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#        
# myData.totalRuns = 10
# myData.maxEvals = 10000
# myData.yTicks = 1000
# myData.graphPadding = 5000
# myData.xAxisLabel = "Algorithm"
# myData.yAxisLabel = "Number of Samples"
# myData.blueLabel = "Fixed NN"
# myData.tanLabel = "Recurrent NN"
# myData.loadData()
# # 
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-24-2020, 09-21-51.txt")
#   
# #MAP ELITES
# myData.data.append([2175, 448, 1321, 818, 169, 1909, 1726, 1080, 2224, 1174])
#      
# #CONTROL  
# myData.data.append([4650, 10000, 2366, 5472, 5469, 2738, 7863, 5422, 10000, 1769])
#  
#  
# myData.loadData("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 01-49-49.txt")
#  
#  
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt")
#   
#   
# myData.data.append([1059, 1280, 3572, 1503, 494, 1138, 1474, 555, 1229, 141])
#  
# #control group RNN
# myData.data.append([232, 1036, 4117, 182, 4314, 3944, 7574, 3577, 3865, 2807])
#  
#  
#   
# testData = myData.displayAlgorithmBoxPlots(["CMA-ES", "N-CMA-ES", "MAP-Elites", "Control", "CMA-ES", "N-CMA-ES", "MAP-Elites", "Control"], "Full Neural Network Results - Easy Race")
#     
# 
# 



 
''' COMBINED MULTI MAZE FNN CODE '''
# myData = DataReporting("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - MultiMaze, 1.0N, FNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 10-48-00.txt")
#       
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#       
# myData.totalRuns = 10
# myData.maxEvals = 1000
# myData.yTicks = 100
# myData.graphPadding = 500
# myData.xAxisLabel = "Algorithm"
# myData.yAxisLabel = "Number of Samples"
# myData.blueLabel = "e-puck"
# myData.tanLabel = "Hemisson"
# myData.loadData()
# # 
#       
# myData.loadData("Algorithm Testing Data/CMA-ES/H-CMAES FULL TEST - MultiMaze, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 12-56-52.txt")
#       
#       
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, FNN, - 10000ME, 50PS, 10TR,  07-29-2020, 10-12-18.txt")
#       
#       
# myData.loadData("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-29-2020, 20-09-36.txt")
#    
# #control FNN epuck
# myData.data.append([125, 76, 17, 9, 247, 173, 72, 113, 84, 58])
#   
# #control FNN hemisson
# myData.data.append([821, 224, 80, 576, 138, 332, 204, 38, 165, 12])
#   
#       
# testData = myData.displayAlgorithmBoxPlots(["CMA-ES", "CMA-ES", "N-CMA-ES", "N-CMA-ES", "Control", "Control"], "Fixed Neural Network Comparison - Multi Maze")
#   
# for i in range(len(myData.data)):
#     for j in range(len(myData.data)):
#         if j <= i:
#             pass
#         else:
#             myData.basicTtest(myData.data[i], myData.data[j]) 
#             print("compared " + str(i) + " to " + str(j))
#          
  
  
#  
 


# ''' COMBINED MULTI MAZE RNN CODE '''
# myData = DataReporting("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - MultiMaze, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 11-01-29.txt")
#     
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#     
# myData.totalRuns = 10
# myData.maxEvals = 1000
# myData.yTicks = 100
# myData.graphPadding = 500
# myData.xAxisLabel = "Algorithm"
# myData.yAxisLabel = "Number of Samples"
# myData.blueLabel = "e-puck"
# myData.tanLabel = "Hemisson"
# myData.loadData()
# # 
#     
# myData.loadData("Algorithm Testing Data/CMA-ES/H-CMAES FULL TEST - MultiMaze, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 07-24-41.txt")
#     
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 10-29-26.txt")
#     
# myData.loadData("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 01-55-53.txt")
#     
# #control e puck RNN
# myData.data.append([13, 202, 7, 23, 470, 7, 96, 232, 22, 23])
#    
# #control hemisson RNN
# myData.data.append([107, 678, 310, 65, 243, 3, 812, 297, 75, 84])
#    
# testData = myData.displayAlgorithmBoxPlots(["CMA-ES", "CMA-ES", "N-CMA-ES", "N-CMA-ES", "Control", "Control"], "Recurrent Neural Network Comparison - Multi Maze")
#     
# print(myData.data)
#    
#   
# for i in range(len(myData.data)):
#     for j in range(len(myData.data)):
#         if j <= i:
#             pass
#         else:
#             myData.basicTtest(myData.data[i], myData.data[j]) 
#             print("compared " + str(i) + " to " + str(j))
#        
# # 



''' COMBINED MULTI MAZE FULL CODE '''
# myData = DataReporting("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - MultiMaze, 1.0N, FNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 10-48-00.txt")
#        
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#        
# myData.totalRuns = 10
# myData.maxEvals = 1000
# myData.yTicks = 100
# myData.graphPadding = 500
# myData.xAxisLabel = "Algorithm/Agent"
# myData.yAxisLabel = "Number of Samples"
# myData.blueLabel = "e-puck"
# myData.tanLabel = "Hemisson"
# myData.loadData()
# # 
#   
# myData.loadData("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - MultiMaze, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 11-01-29.txt")
#   
#        
# myData.loadData("Algorithm Testing Data/CMA-ES/H-CMAES FULL TEST - MultiMaze, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 12-56-52.txt")
#        
# myData.loadData("Algorithm Testing Data/CMA-ES/H-CMAES FULL TEST - MultiMaze, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 07-24-41.txt")
#        
#   
#   
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, FNN, - 10000ME, 50PS, 10TR,  07-29-2020, 10-12-18.txt")
# myData.loadData("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 10-29-26.txt")
#        
#   
#   
# myData.loadData("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-29-2020, 20-09-36.txt")
# myData.loadData("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 01-55-53.txt")
#   
#   
# #control FNN epuck
# myData.data.append([125, 76, 17, 9, 247, 173, 72, 113, 84, 58])
#   
#   
# # #control e puck RNN
# myData.data.append([13, 202, 7, 23, 470, 7, 96, 232, 22, 23])
# #control FNN hemisson
# myData.data.append([821, 224, 80, 576, 138, 332, 204, 38, 165, 12])
#    
# #control hemisson RNN
# myData.data.append([107, 678, 310, 65, 243, 3, 812, 297, 75, 84])
#     
#   
# # #control e puck RNN
# # myData.data.append([13, 202, 7, 23, 470, 7, 96, 232, 22, 23])
# #  
# # #control hemisson RNN
# # myData.data.append([107, 678, 310, 65, 243, 3, 812, 297, 75, 84])
# #  
#   
#   
# testData = myData.displayAlgorithmBoxPlots(["CMA-ES/e-puck", "CMA-ES", "CMA-ES", "CMA-ES", "N-CMA-ES", "N-CMA-ES", "N-CMA-ES", "N-CMA-ES", "Control", "Control", "Control", "Control"], "Full Results - Multi Maze")
#       
#   
# for i in range(len(myData.data)):
#     for j in range(len(myData.data)):
#         if j <= i:
#             pass
#         else:
#             myData.basicTtest(myData.data[i], myData.data[j]) 
#             print("compared " + str(i) + " to " + str(j))
#         
#  
# 

