import matplotlib
matplotlib.use('Qt5Agg')
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
###


def plotLineGraph(data, title, yticks, legend):
    x1 = np.arange(len(data[0]))
    x2 = np.arange(len(data[1]))
    x3 = np.arange(len(data[2]))
    x4 = np.arange(len(data[3]))
    x5 = np.arange(len(data[4]))
    x6 = np.arange(len(data[5]))
    x7 = np.arange(len(data[6]))
    x8 = np.arange(len(data[7]))
    x9 = np.arange(len(data[8]))
    x10 = np.arange(len(data[9]))
    
    
    y1 = data[0]
    y2 = data[1]
    y3 = data[2]
    y4 = data[3]
    y5 = data[4]
    y6 = data[5]
    y7 = data[6]
    y8 = data[7]
    y9 = data[8]
    y10 = data[9]
    
    plt.yticks(np.arange(0, 1, yticks))
    plt.xticks(np.arange(0, len(data[9]), 1))

    plt.plot(x1, y1, label = legend[0])
    # plotting the line 2 points 
    plt.plot(x2, y2, label = legend[1])
    plt.plot(x3, y3, label = legend[2])
    plt.plot(x4, y4, label = legend[3])
    plt.plot(x5, y5, label = legend[4])
    plt.plot(x6, y6, label = legend[5])
    plt.plot(x7, y7, label = legend[6])
    plt.plot(x8, y8, label = legend[7])
    plt.plot(x9, y9, label = legend[8])
    plt.plot(x10, y10, label = legend[9])
    plt.xlabel('Generation', fontsize=14)
    # Set the y axis label of the current axis.
    plt.ylabel('Value', fontsize=14)
    # Set a title of the current axes.
    plt.title(title, fontsize=14)
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()

def convertToDistance(input):
    result = 1.4142*(1-input)
    return result



def generateAVResultsGraphs():
    data=[]
    behaviours=[]
    medianV=[]
    distances=[]
    noveltys=[]
    
    AVERAGE_VELOCITY_TICKS = 0.0025
    MEDIAN_VELOCITY_TICKS = 0.005
    END_POINT_TICKS = 0.025
    NOVELTY_TICKS = 0.0025
    
    myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 3 - 10000ME, 10PS, 1TR,  11-18-2020, 03-44-17.txt") #805
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 9 - 10000ME, 10PS, 1TR,  11-18-2020, 21-09-37.txt")#888
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 6 - 10000ME, 10PS, 1TR,  11-18-2020, 11-42-02.txt") #1076
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 1 - 10000ME, 10PS, 1TR,  11-17-2020, 21-33-21.txt")#1602
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 0 - 10000ME, 10PS, 1TR,  11-17-2020, 20-38-10.txt") #2002
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 5 - 10000ME, 10PS, 1TR,  11-18-2020, 11-04-44.txt") #2726
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 8 - 10000ME, 10PS, 1TR,  11-18-2020, 20-39-09.txt") #5652
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 2 - 10000ME, 10PS, 1TR,  11-18-2020, 03-16-52.txt") #10k
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 4 - 10000ME, 10PS, 1TR,  11-18-2020, 09-31-13.txt") #10k
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 7 - 10000ME, 10PS, 1TR,  11-18-2020, 17-23-42.txt") #10k
    myData.load_graph_data()
    data.append(myData.complex_data)
    
    for i in range(len(data)):
        tempDistances=[]
        tempNoveltys=[]
        tempBehaviours=[]
        tempMedianV=[]
        for j in range(len(data[i][1])):
            tempBehaviours.append(np.max(data[i][1][j]))
            tempMedianV.append(np.median(data[i][1][j]))
            tempDistances.append(convertToDistance(np.max(data[i][2][j])))
            tempNoveltys.append(np.max(data[i][3][j]))
        
        behaviours.append(tempBehaviours)
        medianV.append(tempMedianV)
        distances.append(tempDistances)
        noveltys.append(tempNoveltys)
        
    print(behaviours)
    print(medianV)
    print(distances)
    print(noveltys)
    
    
    legend=["805 sln", "888 sln", "1076 sln", "1602 sln", "2002 sln", "2726 sln", "5652 sln", "10K", "10K", "10K"]
            
    plotLineGraph(behaviours, 'NIP-ES Hard Race Results - Average Velocity descriptor - Max Average Velocity', AVERAGE_VELOCITY_TICKS, legend)
    plotLineGraph(medianV, 'NIP-ES Hard Race Results - Average Velocity descriptor - Median Average Velocity', MEDIAN_VELOCITY_TICKS, legend)
    plotLineGraph(distances, 'NIP-ES Hard Race Results - Average Velocity descriptor - Min Distance from End Point ', END_POINT_TICKS, legend)
    plotLineGraph(noveltys, 'NIP-ES Hard Race Results - Average Velocity descriptor - Max Novelty Score', NOVELTY_TICKS, legend)


def generateMVResultsGraphs():
    data=[]
    behaviours=[]
    medianV=[]
    distances=[]
    noveltys=[]
    
    AVERAGE_VELOCITY_TICKS = 0.0025
    MEDIAN_VELOCITY_TICKS = 0.005
    END_POINT_TICKS = 0.025
    NOVELTY_TICKS = 0.0025
    
    
    myData = DataReporting("(HR) NIPES MEDIANV FULL DATA .90ST 6 - 10000ME, 10PS, 1TR,  12-14-2020, 15-46-03.txt")#560
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES MEDIANV FULL DATA .90ST 3 - 10000ME, 10PS, 1TR,  12-14-2020, 01-17-18.txt") #4243
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES MEDIANV FULL DATA .90ST 9 - 10000ME, 10PS, 1TR,  12-15-2020, 14-58-24.txt") #6668
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES MEDIANV FULL DATA .90ST 0 - 10000ME, 10PS, 1TR,  12-12-2020, 02-06-19.txt")#7416
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES MEDIANV FULL DATA .90ST 4 - 10000ME, 10PS, 1TR,  12-14-2020, 07-37-16.txt") #8254
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES MEDIANV FULL DATA .90ST 7 - 10000ME, 10PS, 1TR,  12-15-2020, 02-04-34.txt") #10k
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES MEDIANV FULL DATA .90ST 1 - 10000ME, 10PS, 1TR,  12-12-2020, 07-48-16.txt")  #10k
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES MEDIANV FULL DATA .90ST 2 - 10000ME, 10PS, 1TR,  12-12-2020, 13-33-32.txt") #10K
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES MEDIANV FULL DATA .90ST 5 - 10000ME, 10PS, 1TR,  12-14-2020, 15-20-39.txt") #10k
    myData.load_graph_data()
    data.append(myData.complex_data)
    myData = DataReporting("(HR) NIPES MEDIANV FULL DATA .90ST 8 - 10000ME, 10PS, 1TR,  12-15-2020, 09-50-37.txt") #10k
    myData.load_graph_data()
    data.append(myData.complex_data)

    
    evals=[]
    for i in range(len(data)):
        tempDistances=[]
        tempNoveltys=[]
        tempBehaviours=[]
        tempMedianV=[]
        evals.append(data[i][0])
        for j in range(len(data[i][1])):
            
            tempBehaviours.append(np.max(data[i][1][j]))
            tempMedianV.append(np.median(data[i][1][j]))
            tempDistances.append(convertToDistance(np.max(data[i][2][j])))
            tempNoveltys.append(np.max(data[i][3][j]))
        
        behaviours.append(tempBehaviours)
        medianV.append(tempMedianV)
        distances.append(tempDistances)
        noveltys.append(tempNoveltys)
        
    
    print(evals)
    
    legend=["560 sln", "4243 sln", "6668 sln", "7416 sln", "8254 sln", "10K", "10K", "10K", "10K", "10K"]
            
    plotLineGraph(behaviours, 'NIP-ES Hard Race Results - Median Velocity descriptor - Max Median Velocity', AVERAGE_VELOCITY_TICKS, legend)
    plotLineGraph(medianV, 'NIP-ES Hard Race Results - Median Velocity descriptor - Median Median Velocity', MEDIAN_VELOCITY_TICKS, legend)
    plotLineGraph(distances, 'NIP-ES Hard Race Results - Median Velocity descriptor - Min Distance from End Point ', END_POINT_TICKS, legend)
    plotLineGraph(noveltys, 'NIP-ES Hard Race Results - Median Velocity descriptor - Max Novelty Score', NOVELTY_TICKS, legend)


generateMVResultsGraphs()
# myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 9 - 10000ME, 10PS, 1TR,  11-18-2020, 21-09-37.txt")#888
#  
# myData.load_graph_data()
# data.append(myData.complex_data)
# tempDistances=[]
# for i in range(len(data[1])):
#     distanceTwo.append(convertToDistance(np.max(data[2][i])))
#     noveltyTwo.append(np.max(data[3][i]))
#     behaviouralTwo.append(np.max(data[1][i]))
#    
#      
# myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 6 - 10000ME, 10PS, 1TR,  11-18-2020, 11-42-02.txt") #1076
# myData.load_graph_data()
# data = myData.complex_data
#  
# for i in range(len(data[1])):
#     distanceThree.append(convertToDistance(np.max(data[2][i])))
#     noveltyThree.append(np.max(data[3][i]))
#     behaviouralThree.append(np.max(data[1][i]))
#      
#  
#  
# myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 1 - 10000ME, 10PS, 1TR,  11-17-2020, 21-33-21.txt")#1602
# myData.load_graph_data()
# data = myData.complex_data
#  
# for i in range(len(data[1])):
#     distanceFour.append(convertToDistance(np.max(data[2][i])))
#     noveltyFour.append(np.max(data[3][i]))
#     behaviouralFour.append(np.max(data[1][i]))
#      
#  
# myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 0 - 10000ME, 10PS, 1TR,  11-17-2020, 20-38-10.txt") #2002
# myData.load_graph_data()
# data = myData.complex_data
#  
# for i in range(len(data[1])):
#     distanceFive.append(convertToDistance(np.max(data[2][i])))
#     noveltyFive.append(np.max(data[3][i]))
#     behaviouralFive.append(np.max(data[1][i]))


    
''' NOVELTY PLOT CODE'''
     
# x1 = np.arange(len(noveltyOne))
# x2 = np.arange(len(noveltyTwo))
# x3 = np.arange(len(noveltyThree))
# x4 = np.arange(len(noveltyFour))
# x5 = np.arange(len(noveltyFive))
# y1 = noveltyOne
# # plotting the line 1 points 
# plt.plot(x1, y1, label = "805 sln")
# # line 2 points
# y2 = noveltyTwo
# y3 = noveltyThree
# y4 = noveltyFour
# y5 = noveltyFive
#  
# #set y ticks
# #        plt.yticks(np.arange(0, self.max_evals + 10, self.y_ticks))
# plt.yticks(np.arange(0, 1, 0.0025))
# plt.xticks(np.arange(0, len(distanceOne), 1))
#  
# # plotting the line 2 points 
# plt.plot(x2, y2, label = "888 sln ")
# plt.plot(x3, y3, label = "1076 sln ")
# plt.plot(x4, y4, label = "1602 sln ")
# plt.plot(x5, y5, label = "2002 sln ")
# plt.xlabel('Generation', fontsize=14)
# # Set the y axis label of the current axis.
# plt.ylabel('Value', fontsize=14)
# # Set a title of the current axes.
# plt.title('NIP-ES Hard Race Results - Average Velocity descriptor - Max Novelty Score')
# # show a legend on the plot
# plt.legend()
# # Display a figure.
# plt.show()









'''DISTANCE FROM END POINT CODE ''' 
# 
# def convertToDistance(input):
#     result = 1.4142*(1-input)
#     return result
#  
# distanceOne = []
# distanceTwo= []
# distanceThree=[]
# distanceFour=[]
# distanceFive=[]
#  
# noveltyOne=[]
# noveltyTwo=[]
# noveltyThree=[]
# noveltyFour=[]
# noveltyFive=[]
#  
# behaviouralOne=[]
# behaviouralTwo=[]
# behaviouralThree=[]
# behaviouralFour=[]
# behaviouralFive=[]
#  
#  
# myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 3 - 10000ME, 10PS, 1TR,  11-18-2020, 03-44-17.txt") #805
# #myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 9 - 10000ME, 10PS, 1TR,  11-18-2020, 21-09-37.txt")#888
# #myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 6 - 10000ME, 10PS, 1TR,  11-18-2020, 11-42-02.txt") #1076
# #myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 1 - 10000ME, 10PS, 1TR,  11-17-2020, 21-33-21.txt")#1602
# #myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 0 - 10000ME, 10PS, 1TR,  11-17-2020, 20-38-10.txt") #2002
#  
#  
#  
# myData.load_graph_data()
# #temp_sln_evals, temp_behavioural_descriptors, temp_objective_fitnesses, temp_novelty_scores
# #print(myData.complex_data)
# data = myData.complex_data
#  
# for i in range(len(data[1])):
#     distanceOne.append(convertToDistance(np.max(data[2][i])))
#     noveltyOne.append(np.max(data[3][i]))
#     behaviouralOne.append(np.max(data[1][i]))
#      
#      
# myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 9 - 10000ME, 10PS, 1TR,  11-18-2020, 21-09-37.txt")
#  
# myData.load_graph_data()
# data = myData.complex_data
#  
# for i in range(len(data[1])):
#     distanceTwo.append(convertToDistance(np.max(data[2][i])))
#     noveltyTwo.append(np.max(data[3][i]))
#     behaviouralTwo.append(np.max(data[1][i]))
#    
#      
# myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 6 - 10000ME, 10PS, 1TR,  11-18-2020, 11-42-02.txt") #1076
# myData.load_graph_data()
# data = myData.complex_data
#  
# for i in range(len(data[1])):
#     distanceThree.append(convertToDistance(np.max(data[2][i])))
#     noveltyThree.append(np.max(data[3][i]))
#     behaviouralThree.append(np.max(data[1][i]))
#      
#  
#  
# myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 1 - 10000ME, 10PS, 1TR,  11-17-2020, 21-33-21.txt")#1602
# myData.load_graph_data()
# data = myData.complex_data
#  
# for i in range(len(data[1])):
#     distanceFour.append(convertToDistance(np.max(data[2][i])))
#     noveltyFour.append(np.max(data[3][i]))
#     behaviouralFour.append(np.max(data[1][i]))
#      
#  
# myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 0 - 10000ME, 10PS, 1TR,  11-17-2020, 20-38-10.txt") #2002
# myData.load_graph_data()
# data = myData.complex_data
#  
# for i in range(len(data[1])):
#     distanceFive.append(convertToDistance(np.max(data[2][i])))
#     noveltyFive.append(np.max(data[3][i]))
#     behaviouralFive.append(np.max(data[1][i]))
#      
#      
#      
# x1 = np.arange(len(distanceOne))
# x2 = np.arange(len(distanceTwo))
# x3 = np.arange(len(distanceThree))
# x4 = np.arange(len(distanceFour))
# x5 = np.arange(len(distanceFive))
# y1 = distanceOne
# # plotting the line 1 points 
# plt.plot(x1, y1, label = "805 sln (cm)")
# # line 2 points
# y2 = distanceTwo
# y3 = distanceThree
# y4 = distanceFour
# y5 = distanceFive
#  
# #set y ticks
# #        plt.yticks(np.arange(0, self.max_evals + 10, self.y_ticks))
# plt.yticks(np.arange(0, 1, 0.025))
# plt.xticks(np.arange(0, len(distanceOne), 1))
#  
# # plotting the line 2 points 
# plt.plot(x2, y2, label = "888 sln (cm)")
# plt.plot(x3, y3, label = "1076 sln (cm)")
# plt.plot(x4, y4, label = "1602 sln (cm)")
# plt.plot(x5, y5, label = "2002 sln (cm)")
# plt.xlabel('Generation', fontsize=14)
# # Set the y axis label of the current axis.
# plt.ylabel('Value', fontsize=14)
# # Set a title of the current axes.
# plt.title('NIP-ES Hard Race Results - Average Velocity descriptor - Min Distance from End Point')
# # show a legend on the plot
# plt.legend()
# # Display a figure.
# plt.show()

# myData = DataReporting("(HR) NIPES AV FULL DATA .90ST 3 - 10000ME, 10PS, 1TR,  11-18-2020, 03-44-17.txt")
# myData.load_graph_data()
# #temp_sln_evals, temp_behavioural_descriptors, temp_objective_fitnesses, temp_novelty_scores
# #print(myData.complex_data)
# data = myData.complex_data
# 
# count = 0
# objective_maxes=[]
# novelty_maxes=[]
# 
# for i in range(len(data[1])):
#     objective_maxes.append(np.max(data[2][i]))
#     novelty_maxes.append(np.max(data[3][i]))
#         
# print(objective_maxes)
# print(novelty_maxes)
# 
# ''' DATA RESULTS VARIANCE BOX PLOT CODE '''
# myData = DataReporting()
#   
#  
# myData.data.append(objective_maxes)  
# 
# myData.data.append(novelty_maxes)  
# 
# 
# myData.total_runs = 10
# myData.max_evals = 1
# myData.y_ticks = 0.1
# myData.graph_padding = 0
# myData.x_axis_label = "Novelty Behavioural Descriptor"
# myData.y_axis_label = "Number of Samples"
#   
# testData = myData.display_algorithm_box_plots(["Objective Fitnesses", "Novelty Scores"], "Data Results Box Plots - Variance of Objective Fitness and Novelty Score")




#so plot generation on bottom of graph which is index in the array and then repeat code for the novelty score and plot line graph
#print(data)
#temp = data[0][0]
#for i in range(len(data)):
#    for j in range(len(data[i])):
#        if(data[i][j] > temp):
#            temp = data[i]
#
#print("highest = " + str(temp))


#average velocity RNN results:
#[-1, -1, 34, -1, 167, 425, -1, -1, -1, 31]
# DFS DATA:
#[-1, 33, 1429, 2875, -1, 129, 1294, 2115, -1, -1]

#regular novelty data
#[144, 2412, 514, 806, 13, 3040, 180, 3736, 2297, 1466]

#MAP=Elites results
#[1221, 1064, 1078, 1356, 397, 3753, 1102, 67, 3647, 2102]

#NIPES results
#[2016, 1666, 1364, 493, 963, 387, 544, 3505, 5, 2030]

#NIPES AV RESULTS
#[969, 287, 893, 656, 835, 545, 1051, 18, 219, 1540]

#NIPES DFS RESULTS:
#[188, 6134, 367, 1427, 3120, 146, 861, 1791, 1136, 81]



#NIPES HARDRACE RNDPOINTS
# [10000, 6249, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]

#NIPES AV HARDRACE
#[8214, 10000, 10000, 10000, 10000, 10000, 6627, 2566, 10000, 10000]

#NIPES DFS HARDRACE
#[10000, 10000, 10000, 7786, 3162, 10000, 10000, 1114, 10000, 7497]

#NEW NIPES AV TESTING HARD RACE 0.9 SLN THRESHOLD
#[2002, 1602, 10000, 805, 10000, 2726, 1076, 10000, 5652, 888]
#NEW NIPES DFS TESTING HARD RACE 0.9 SLN THRESHOLD
#[2888, 8553, 7585, 10000, 10000, 7565, 8502, 10000, 10000, 10000]
#NEW NIPES ENDPOINTS TESTING HARD RACE 0.9 SLN THRESHOLD
#[10000, 10000, 10000, 10000, 1897, 10000, 1980, 10000, 10000, 7517]
#NEW NIPES MEDIAN V TESTING HARD RACE 0.9 SLN THRESHOLD
#[8990, 10000, 10000, 10000, 9463, 5565, 8387, 10000, 10000, 4932, 2952, 10000]
#CORRECTED NEW NIPES MEDIAN V TESTING HARD RACE 0.9 SLN THRESHOLD
#[4243, 8254, 10000, 560, 10000, 10000, 6668, 7416, 10000, 10000]

##NEW NIPES AV TESTING HARD RACE 0.9 SLN THRESHOLD MULTI DESCRIPTOR
#AV & DFS
#[3063, 7424, 10000, 10000, 4411, 8344, 10000, 10000, 938, 10000]

#NEW MAP ELITES HARD RACE 0.9 AV AND DFS
#[10000,10000,7150,10000,2121,3704,6313,10000,10000,10000]

'''Additional Experimentation Code'''
#AV
#print("myData.complex_data")

'''HARD RACE 0.9 GRAPH CODE  '''
# myData = DataReporting()
#  
# #ORDER
# #SLN EVALS
# #BEHAVIOURAL
# #OBJECTIVE FITNESSES
# #NOVELTY SCORES
#  
#  
# #AV
# myData.data.append([2002, 1602, 10000, 805, 10000, 2726, 1076, 10000, 5652, 888])  
# #MV    
# myData.data.append([4243, 8254, 10000, 560, 10000, 10000, 6668, 7416, 10000, 10000])      
# #DFS
# myData.data.append([2888, 8553, 7585, 10000, 10000, 7565, 8502, 10000, 10000, 10000])  
# #ENDPOINTS    
# myData.data.append([10000, 10000, 10000, 10000, 1897, 10000, 1980, 10000, 10000, 7517])      
#  
#  
# #myData.load_data()
# #DFS
# #myData.data.append([-1, 33, 1429, 2875, -1, 129, 1294, 2115, -1, -1])      
#   
# #ENDPOINTS
# #myData.data.append([144, 2412, 514, 806, 13, 3040, 180, 3736, 2297, 1466])      
#    
# #MAP-Elites
# #myData.data.append([1221, 1064, 1078, 1356, 397, 3753, 1102, 67, 3647, 2102])      
#    
# myData.total_runs = 10
# myData.max_evals = 10000
# myData.y_ticks = 1000
# myData.graph_padding = 2000
# myData.x_axis_label = "Novelty Behavioural Descriptor"
# myData.y_axis_label = "Number of Samples"
#   
# testData = myData.display_algorithm_box_plots(["Average Velocity", "Median Velocity", "Distance FS", "Endpoints"], "NIP-ES Single Descriptor Results - Hard Race 0.9 sln threshold")



#'''HARD RACE 0.9 MULTI DESCRIPTOR GRAPH CODE  '''
#myData = DataReporting()
# 
##ORDER
##SLN EVALS
##BEHAVIOURAL
##OBJECTIVE FITNESSES
##NOVELTY SCORES
# 
# 
##AV & DFS
#myData.data.append([3063, 7424, 10000, 10000, 4411, 8344, 10000, 10000, 938, 10000])
##MAP ELITES
#myData.data.append([10000,10000,7150,10000,2121,3704,6313,10000,10000,10000])        
# 
# 
##myData.load_data()
##DFS
##myData.data.append([-1, 33, 1429, 2875, -1, 129, 1294, 2115, -1, -1])      
#  
##ENDPOINTS
##myData.data.append([144, 2412, 514, 806, 13, 3040, 180, 3736, 2297, 1466])      
#   
##MAP-Elites
##myData.data.append([1221, 1064, 1078, 1356, 397, 3753, 1102, 67, 3647, 2102])      
#   
#myData.total_runs = 10
#myData.max_evals = 10000
#myData.y_ticks = 1000
#myData.graph_padding = 2000
#myData.x_axis_label = "Algorithm/Novelty Behavioural Descriptors"
#myData.y_axis_label = "Number of Samples"
#  
#testData = myData.display_algorithm_box_plots(["NIP-ES - AV/DFS", "MAP-Elites - AV/DFS"], "Multi-Descriptor Algorithm Results - Hard Race 0.9 sln threshold")

''' END OF MULTI DESCRIPTOR GRAPH CODE '''

'''NIPES Experimentation Code'''
#AV
#myData = DataReporting()
#       
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#
#AV
#myData.data.append([-1, -1, 34, -1, 167, 425, -1, -1, -1, 31])      
#myData.load_data()
#DFS
#NIPES results
#[2016, 1666, 1364, 493, 963, 387, 544, 3505, 5, 2030]

#NIPES AV RESULTS
#[969, 287, 893, 656, 835, 545, 1051, 18, 219, 1540]

#NIPES DFS RESULTS:
#[188, 6134, 367, 1427, 3120, 146, 861, 1791, 1136, 81]
  

# #KNN
# myData.data.append([969, 287, 893, 656, 835, 545, 1051, 18, 219, 1540])      
# 
# #MAP-Elites
# myData.data.append([188, 6134, 367, 1427, 3120, 146, 861, 1791, 1136, 81])      
# 
# myData.data.append([2016, 1666, 1364, 493, 963, 387, 544, 3505, 5, 2030])    
# 
# myData.total_runs = 10
# myData.max_evals = 10000
# myData.y_ticks = 1000
# myData.graph_padding = 5000
# myData.x_axis_label = "Novelty Behavioural Descriptor"
# myData.y_axis_label = "Number of Samples"
# # 
#   
# 
# #MAP ELITES
#      
# #CONTROL  
#   
# testData = myData.display_algorithm_box_plots(["Average Velocity", "Distance FS", "Endpoints"], "NIP-ES Comparison - Behavioural Descriptors Results")
#     


'''FULL NIPES AND NCMAES RESULTS Experimentation Code'''
#AV

# myData = DataReporting("NCMAES AV RUN 9 - 10000ME, 50PS, 1TR,  10-06-2020, 04-15-21.txt")
# #       
# # '''Manually ensure the DataReporing class is set with relevant parameters ''' 
# #
# #AV
# #myData.data.append([-1, -1, 34, -1, 167, 425, -1, -1, -1, 31])      
# myData.load_data()
# 
# #NIPES AV
# myData.data.append([969, 287, 893, 656, 835, 545, 1051, 18, 219, 1540])      
# #DFS
# myData.data.append([-1, 33, 1429, 2875, -1, 129, 1294, 2115, -1, -1])    
# #NIPES DFS
# myData.data.append([188, 6134, 367, 1427, 3120, 146, 861, 1791, 1136, 81])      
# 
#  
# #ENDPOINTS
# myData.data.append([144, 2412, 514, 806, 13, 3040, 180, 3736, 2297, 1466])      
# 
# #NIPES ENDPOINTS
# myData.data.append([2016, 1666, 1364, 493, 963, 387, 544, 3505, 5, 2030])    
# #MAP-Elites
# #myData.data.append([1221, 1064, 1078, 1356, 397, 3753, 1102, 67, 3647, 2102])      
#  
# 
# 
# 
# 
# 
# myData.total_runs = 10
# myData.max_evals = 10000
# myData.y_ticks = 1000
# myData.graph_padding = 5000
# myData.x_axis_label = "Novelty Behavioural Descriptor"
# myData.y_axis_label = "Number of Samples"
# myData.blue_label = "N-CMA-ES"
# myData.tan_label = "NIP-ES"
# 
# testData = myData.display_algorithm_box_plots(["Average Velocity", "Average Velocity", "Distance FS", "Distance FS", "Endpoints", "Endpoints"], "NIP-ES/N-CMA-ES Comparison - Behavioural Descriptors Results")


'''BOXPLOT Exmaple from Sigma test data'''
# myData = DataReporting("Sigma Test - EscapeRoom - 150ME, 25PS, 25SR, 0.1-2.0, 07-21-2020, 12-17-57.txt")
# #myData = DataReporting("NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt")
#  '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#  #myData.upper_limit = 2.0
#  #myData.lower_limit = 0.1
#  
# myData.total_runs = 10
# myData.max_evals = 10000
# myData.y_ticks = 1000
# myData.load_data()
# 
# myData.fullResults = [248, 2707, 5613, 5510, 13, 3354, 206, 802, -1, 1225]
# file = open("NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt",'wb')
# pickle.dump(myData.fullResults, file)
# file.close()
# 
# testData = myData.displaySigmaBoxPlots()
# testData = myData.display_algorithm_box_plots("NCMA-ES")
# 
# [-1, 2707, 5613, 5510, 13, 3354, 206, 802, -1, 1225]

'''BOXPLOT Exmaple from Algorithm Test Data'''
#myData = DataReporting("Algorithm Testing Data/CMA-ES/CMA-ES TEST - MiddleWall - 150ME, 25PS, 20TR, 07-21-2020, 00-28-46.txt")
#myData.total_runs = 20
#myData.max_evals = 150
#myData.load_data()
#myData.display_algorithm_box_plots("CMA-ES", "NCMA-ES")









# myData = DataReporting("Algorithm Testing Data/SEND-THIS-TO-DREW-H-CMAES-FNN-Easy-Race-0-10000ME-50PS-1TR-08-03-2020-17-47-53.txt")
# myData.load_data()
# print(myData.data)
''' HEATMAP FOR EASY RACE E PUCK FNN'''

# def myfunc():
#     pass
#  
# myMap = MAPElites(58, myfunc, bin_count = 10, tournament_size = 5, max_velocity = 0.125, max_distance = 1.2)
#  
# myMap.load_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 9 5TS, 10BC, 10000gens08-05-2020, 02-47-57.txt")
#  
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 8 5TS, 10BC, 10000gens08-05-2020, 01-55-06.txt")
#  
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 7 5TS, 10BC, 10000gens08-05-2020, 00-05-48.txt")
#  
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 6 5TS, 10BC, 10000gens08-04-2020, 23-13-05.txt")
#  
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 5 5TS, 10BC, 10000gens08-04-2020, 21-48-43.txt")
#  
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 4 5TS, 10BC, 10000gens08-04-2020, 20-13-12.txt")
#  
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 3 5TS, 10BC, 10000gens08-04-2020, 20-04-07.txt")
#  
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 2 5TS, 10BC, 10000gens08-04-2020, 19-17-57.txt")
#  
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 1 5TS, 10BC, 10000gens08-04-2020, 18-09-00.txt")
#  
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/FNN/FINAL/MAPElites easy race FNN 1KIP 0 5TS, 10BC, 10000gens08-04-2020, 17-45-56.txt")
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
# myData.total_runs = 10
# myData.max_evals = 10000
# myData.y_ticks = 1000
# myData.graph_padding = 1000
# myData.x_axis_label = "Agent/Neural Network"
# myData.y_axis_label = "Number of Samples"
#  
#  
# testData = myData.display_algorithm_box_plots("MAP-Elites - Easy Race", "e-puck FNN", "e-puck RNN", "Hemisson FNN", "Hemisson RNN")
# 
# 
#  
''' HEATMAP FOR EASY RACE E PUCK RNN'''
 
# def myfunc():
#     pass
# 
# myMap = MAPElites(122, myfunc, bin_count = 10, tournament_size = 5, max_velocity = 0.125, max_distance = 1.2)
# 
# myMap.load_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 9 5TS, 10BC, 10000gens08-05-2020, 02-33-59.txt")
# 
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 8 5TS, 10BC, 10000gens08-05-2020, 02-27-00.txt")
# 
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 7 5TS, 10BC, 10000gens08-05-2020, 01-25-57.txt")
# 
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 6 5TS, 10BC, 10000gens08-05-2020, 00-58-13.txt")
# 
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 5 5TS, 10BC, 10000gens08-04-2020, 23-45-28.txt")
# 
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 4 5TS, 10BC, 10000gens08-04-2020, 22-49-12.txt")
# 
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 3 5TS, 10BC, 10000gens08-04-2020, 22-24-43.txt")
# 
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 2 5TS, 10BC, 10000gens08-04-2020, 21-08-37.txt")
# 
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 1 5TS, 10BC, 10000gens08-04-2020, 17-57-06.txt")
# 
# myMap.combine_loaded_map("Algorithm Testing Data/MAP-Elites/eASY RACE TESTING/RNN/Final/MAPElites easy race RNN 1KIP 0 5TS, 10BC, 10000gens08-04-2020, 16-48-35.txt")
# 
# 
# myMap.heatmap_title = "MAP-Elites Heatmap - Easy Race - Custom e-puck - RNN" 
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
# myMap.generate_heatmap()



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
# myData.load_data()
# print(myData.data)

#12 + 24 +  

''' Sigma significance testing line graph code'''

#Escape room: ({1.4: 7, 1.3: 4, 0.6: 3, 1.0: 2, 1.9: 2, 0.2: 1, 0.5: 1, 0.8: 1, 0.9: 1, 1.2: 1, 1.7: 1})
#Middle Wall: ({1.7: 7, 1.3: 5, 1.9: 5, 0.5: 5, 1.8: 5, 2.0: 5, 0.3: 4, 1.0: 4, 1.2: 3, 1.6: 3, 0.4: 1, 0.6: 1})
#Multi Maze: ({2.0: 6, 0.3: 2, 1.7: 1, 0.6: 1, 0.7: 1, 1.3: 1, 1.4: 1, 1.8: 1})


# myData = DataReporting()
# #myData.load_data()
# myData.upper_limit = 2.0
# myData.lower_limit = 0.1
# myData.max_evals = 65
# myData.y_ticks = 5
# myData.x_axis_label = "Sigma Value"
# myData.y_axis_label = "Total Null Hypothesis Rejections"
# #myData.data = [0.1:0, 0.2:1, 0.3:6, 0.4:1, 0.5:6, 0.6:5, 0.7:1, 0.8:1, 0.9:1, 1.0:4, 1.1:0, 1.2:4, 1.3:10, 1.4:8, 1.5:0, 1.6:3, 1.7:9, 1.8:1, 1.9:5, 2.0:11]
# myData.data = [0, 1, 6, 1, 6, 5, 1, 1, 1, 4, 0, 4, 10, 8, 0, 3, 9, 1, 5, 11]
# 
# 
# myData.display_line_graph("Sigma Significance Testing (p value and Student's T Testing)")

''' sigma clipping graph'''
# 
 
 
# myData = DataReporting()
# #myData.load_data()
# myData.upper_limit = 2.0
# myData.lower_limit = 0.1
# myData.max_evals = 100
# myData.y_ticks = 10
# myData.x_axis_label = "Sigma Value"
# myData.y_axis_label = "Percentage of Avg 'Clipped' Genes"
# myData.data = [0.0, 0.0, 0.09, 1.04, 4.69, 9.7, 15.1, 21.31, 26.8, 30.85, 36.49, 40.25, 45.33, 48.2, 50.57, 52.36, 55.54, 57.8, 59.62, 61.46]
#  
#  
# myData.displayLineGraph("Proportion of 'Clipped' Genes per Individual in Initial CMA-ES Pop")


# #tempEA.saveFile = "EasyRace initial test DFS 1K TS5.txt"
# #tempEA.load_map("EasyRace initial test DFS 1K TS5.txt")
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
#myData.upper_limit = 2.0
#myData.lower_limit = 0.1
# 
# myData.total_runs = 10
# myData.max_evals = 550
# myData.y_ticks = 100
# myData.graph_padding = 10
# myData.x_axis_label = "Agent/Neural Network"
# myData.y_axis_label = "Number of Samples"
# myData.load_data()
# # 
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 10-29-26.txt")
# 
# 
# myData.load_data("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-29-2020, 20-09-36.txt")
# 
# myData.load_data("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 01-55-53.txt")

#myData.load_data("Algorithm Testing Data/CMA-ES/H-CMAES FULL TEST - EscapeRoom, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 20-17-18.txt")

#myData.fullResults = [248, 2707, 5613, 5510, 13, 3354, 206, 802, -1, 1225]
#file = open("NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt",'wb')
#pickle.dump(myData.fullResults, file)
#file.close()





#testData = myData.displaySigmaBoxPlots("Sigma Testing - Escape Room")
#testData = myData.display_algorithm_box_plots("N-CMA-ES TEST - Multi Maze", "e-puck FNN", "e-puck RNN", "Hemisson FNN", "Hemisson RNN")

#[-1, 2707, 5613, 5510, 13, 3354, 206, 802, -1, 1225]



'''SIGMA CODE '''
# myData = DataReporting("Sigma Testing Data/Sigma Test - EscapeRoom - 150ME, 25PS, 25SR, 0.1-2.0, 07-21-2020, 12-17-57.txt")
# #myData = DataReporting("Sigma Testing Data/Sigma Test - MiddleWall - 150ME, 25PS, 25SR, 0.1-2.0, 07-21-2020, 00-28-46.txt")
# #myData = DataReporting("Sigma Testing Data/Sigma Test - MultiMaze - 300ME, 50PS, 25SR, 0.1-2.0, 07-23-2020, 03-46-21.txt")
#    
# myData.upper_limit = 2.0
# myData.lower_limit = 0.1
#  
# myData.total_runs = 25
# myData.max_evals = 300
# myData.y_ticks = 100
# myData.x_axis_label = "Sigma Value"
# myData.y_axis_label = "Number of Samples"
# myData.graph_padding = 20
# myData.load_data()
# testData = myData.displaySigmaBoxPlots("Sigma Testing - Middle Wall")
'''BOXPLOT Exmaple from Algorithm Test Data'''
#myData = DataReporting("Algorithm Testing Data/CMA-ES/CMA-ES TEST - MiddleWall - 150ME, 25PS, 20TR, 07-21-2020, 00-28-46.txt")
#myData.total_runs = 20
#myData.max_evals = 150
#myData.load_data()
#myData.display_algorithm_box_plots("CMA-ES", "NCMA-ES")




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

# myData.total_runs = 10
# myData.max_evals = 550
# myData.y_ticks = 100
# myData.graph_padding = 10
# myData.x_axis_label = "Agent/Neural Network"
# myData.y_axis_label = "Number of Samples"
# myData.load_data()
# # 
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 10-29-26.txt")
# 
# 
# myData.load_data("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-29-2020, 20-09-36.txt")
# 
# myData.load_data("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 01-55-53.txt")
# 
# 
# testData = myData.display_algorithm_box_plots("N-CMA-ES TEST - Multi Maze", "e-puck FNN", "e-puck RNN", "Hemisson FNN", "Hemisson RNN")
# 
''' NCMAES EASY RACE CODE '''
 
# myData = DataReporting("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-24-2020, 09-21-51.txt")
#  
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#  
# myData.total_runs = 10
# myData.max_evals = 10000
# myData.y_ticks = 1000
# myData.graph_padding = 1000
# myData.x_axis_label = "Agent/Neural Network"
# myData.y_axis_label = "Number of Samples"
# myData.load_data()
# # 
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt")
# 
# myData.load_data("Algorithm Testing Data/NCMA-ES/FOR-TOM-SEND-2-DREW-1-HEMISSON-EASY-RACE-RNN-10000ME-50PS-1TR-07-30-2020-11-08-54.txt")
# myData.load_data("Algorithm Testing Data/NCMA-ES/FOR-TOM-SEND-2-DREW-1-HEMISSON-EASY-RACE-RNN-10000ME-50PS-1TR-07-30-2020-11-08-54.txt")
# 
# testData = myData.display_algorithm_box_plots("N-CMA-ES TEST - Easy Race", "e-puck FNN", "e-puck RNN", "Hemisson FNN", "Hemisson RNN")



''' NCMAES Escape Room CODE '''
# myData = DataReporting("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EscapeRoom, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-29-2020, 12-37-46.txt")
# 
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
# 
# myData.total_runs = 10
# myData.max_evals = 30
# myData.y_ticks = 5
# myData.graph_padding = 10
# myData.x_axis_label = "Agent/Neural Network"
# myData.y_axis_label = "Number of Samples"
# myData.load_data()
# # 
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EscapeRoom, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-29-2020, 12-45-20.txt")
#  
#  
# myData.load_data("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - EscapeRoom, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 19-36-28.txt")
#  
# myData.load_data("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - EscapeRoom, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 20-00-33.txt")
#  
#  
# testData = myData.display_algorithm_box_plots("N-CMA-ES TEST - Escape Room", "e-puck FNN", "e-puck RNN", "Hemisson FNN", "Hemisson RNN")


''' COMBINED EASY RACE CODE '''
    
# myData = DataReporting("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-25-2020, 22-38-31.txt")
#            
# myData.total_runs = 10
# myData.max_evals = 10000
# myData.y_ticks = 1000
# myData.graph_padding = 5000
# myData.x_axis_label = "Algorithm/Neural Network"
# myData.y_axis_label = "Number of Samples"
# myData.blue_label = "Fixed NN"
# myData.tan_label = "Recurrent NN"
# myData.load_data()
#     
# myData.load_data("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 01-49-49.txt")
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-24-2020, 09-21-51.txt")
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt")
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
# testData = myData.display_algorithm_box_plots(["CMA-ES", " CMA-ES", "N-CMA-ES", "N-CMA-ES", "MAP-Elites", "MAP-Elites", "Control", "Control"], "Algorithms Comparison - Easy Race - e-puck only")
#  


''' COMBINED EASY RACE RNN CODE with control group '''
# #  
# myData = DataReporting("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 01-49-49.txt")
#       
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#       
# myData.total_runs = 10
# myData.max_evals = 10000
# myData.y_ticks = 1000
# myData.graph_padding = 5000
# myData.x_axis_label = "Algorithm"
# myData.y_axis_label = "Number of Samples"
# myData.load_data()
# # 
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt")
#     
#     
# myData.data.append([1059, 1280, 3572, 1503, 494, 1138, 1474, 555, 1229, 141])
#    
# #control group RNN
# myData.data.append([232, 1036, 4117, 182, 4314, 3944, 7574, 3577, 3865, 2807])
#    
#    
# testData = myData.display_algorithm_box_plots(["CMA-ES", "N-CMA-ES", "MAP-Elites", "Control"], "Recurrent Neural Network Results - Easy Race")



''' COMBINED EASY RACE FNN CODE'''
#   
# myData = DataReporting("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-25-2020, 22-38-31.txt")
#       
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#       
# myData.total_runs = 10
# myData.max_evals = 10000
# myData.y_ticks = 1000
# myData.graph_padding = 5000
# myData.x_axis_label = "Algorithm"
# myData.y_axis_label = "Number of Samples"
# myData.load_data()
# # 
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-24-2020, 09-21-51.txt")
#  
# #MAP ELITES
# myData.data.append([2175, 448, 1321, 818, 169, 1909, 1726, 1080, 2224, 1174])
#     
# #CONTROL  
# myData.data.append([4650, 10000, 2366, 5472, 5469, 2738, 7863, 5422, 10000, 1769])
#  
# testData = myData.display_algorithm_box_plots(["CMA-ES", "N-CMA-ES", "MAP-Elites", "Control"], "Fixed Neural Network Results - Easy Race")
#    
# 



''' COMBINED EASY RACE FULL NN CODE'''
   
# myData = DataReporting("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-25-2020, 22-38-31.txt")
#        
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#        
# myData.total_runs = 10
# myData.max_evals = 10000
# myData.y_ticks = 1000
# myData.graph_padding = 5000
# myData.x_axis_label = "Algorithm"
# myData.y_axis_label = "Number of Samples"
# myData.blue_label = "Fixed NN"
# myData.tan_label = "Recurrent NN"
# myData.load_data()
# # 
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, FNN,   - 100ME, 50PS, 10TR,  07-24-2020, 09-21-51.txt")
#   
# #MAP ELITES
# myData.data.append([2175, 448, 1321, 818, 169, 1909, 1726, 1080, 2224, 1174])
#      
# #CONTROL  
# myData.data.append([4650, 10000, 2366, 5472, 5469, 2738, 7863, 5422, 10000, 1769])
#  
#  
# myData.load_data("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 01-49-49.txt")
#  
#  
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - EasyRace, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-27-2020, 09-37-54.txt")
#   
#   
# myData.data.append([1059, 1280, 3572, 1503, 494, 1138, 1474, 555, 1229, 141])
#  
# #control group RNN
# myData.data.append([232, 1036, 4117, 182, 4314, 3944, 7574, 3577, 3865, 2807])
#  
#  
#   
# testData = myData.display_algorithm_box_plots(["CMA-ES", "N-CMA-ES", "MAP-Elites", "Control", "CMA-ES", "N-CMA-ES", "MAP-Elites", "Control"], "Full Neural Network Results - Easy Race")
#     
# 
# 



 
''' COMBINED MULTI MAZE FNN CODE '''
# myData = DataReporting("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - MultiMaze, 1.0N, FNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 10-48-00.txt")
#       
# '''Manually ensure the DataReporing class is set with relevant parameters ''' 
#       
# myData.total_runs = 10
# myData.max_evals = 1000
# myData.y_ticks = 100
# myData.graph_padding = 500
# myData.x_axis_label = "Algorithm"
# myData.y_axis_label = "Number of Samples"
# myData.blue_label = "e-puck"
# myData.tan_label = "Hemisson"
# myData.load_data()
# # 
#       
# myData.load_data("Algorithm Testing Data/CMA-ES/H-CMAES FULL TEST - MultiMaze, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 12-56-52.txt")
#       
#       
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, FNN, - 10000ME, 50PS, 10TR,  07-29-2020, 10-12-18.txt")
#       
#       
# myData.load_data("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-29-2020, 20-09-36.txt")
#    
# #control FNN epuck
# myData.data.append([125, 76, 17, 9, 247, 173, 72, 113, 84, 58])
#   
# #control FNN hemisson
# myData.data.append([821, 224, 80, 576, 138, 332, 204, 38, 165, 12])
#   
#       
# testData = myData.display_algorithm_box_plots(["CMA-ES", "CMA-ES", "N-CMA-ES", "N-CMA-ES", "Control", "Control"], "Fixed Neural Network Comparison - Multi Maze")
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
# myData.total_runs = 10
# myData.max_evals = 1000
# myData.y_ticks = 100
# myData.graph_padding = 500
# myData.x_axis_label = "Algorithm"
# myData.y_axis_label = "Number of Samples"
# myData.blue_label = "e-puck"
# myData.tan_label = "Hemisson"
# myData.load_data()
# # 
#     
# myData.load_data("Algorithm Testing Data/CMA-ES/H-CMAES FULL TEST - MultiMaze, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 07-24-41.txt")
#     
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 10-29-26.txt")
#     
# myData.load_data("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 01-55-53.txt")
#     
# #control e puck RNN
# myData.data.append([13, 202, 7, 23, 470, 7, 96, 232, 22, 23])
#    
# #control hemisson RNN
# myData.data.append([107, 678, 310, 65, 243, 3, 812, 297, 75, 84])
#    
# testData = myData.display_algorithm_box_plots(["CMA-ES", "CMA-ES", "N-CMA-ES", "N-CMA-ES", "Control", "Control"], "Recurrent Neural Network Comparison - Multi Maze")
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
# myData.total_runs = 10
# myData.max_evals = 1000
# myData.y_ticks = 100
# myData.graph_padding = 500
# myData.x_axis_label = "Algorithm/Agent"
# myData.y_axis_label = "Number of Samples"
# myData.blue_label = "e-puck"
# myData.tan_label = "Hemisson"
# myData.load_data()
# # 
#   
# myData.load_data("Algorithm Testing Data/CMA-ES/e-puck/CMAES FULL TEST - MultiMaze, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 11-01-29.txt")
#   
#        
# myData.load_data("Algorithm Testing Data/CMA-ES/H-CMAES FULL TEST - MultiMaze, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 12-56-52.txt")
#        
# myData.load_data("Algorithm Testing Data/CMA-ES/H-CMAES FULL TEST - MultiMaze, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 07-24-41.txt")
#        
#   
#   
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, FNN, - 10000ME, 50PS, 10TR,  07-29-2020, 10-12-18.txt")
# myData.load_data("Algorithm Testing Data/NCMA-ES/e-puck/NCMAES FULL TEST - MultiMaze, 1.0N, RNN,   - 10000ME, 50PS, 10TR,  07-29-2020, 10-29-26.txt")
#        
#   
#   
# myData.load_data("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, FNN,  - 10000ME, 50PS, 10TR,  07-29-2020, 20-09-36.txt")
# myData.load_data("Algorithm Testing Data/NCMA-ES/H-NCMAES FULL TEST - MultiMaze, 1.0N, RNN,  - 10000ME, 50PS, 10TR,  07-30-2020, 01-55-53.txt")
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
# testData = myData.display_algorithm_box_plots(["CMA-ES/e-puck", "CMA-ES", "CMA-ES", "CMA-ES", "N-CMA-ES", "N-CMA-ES", "N-CMA-ES", "N-CMA-ES", "Control", "Control", "Control", "Control"], "Full Results - Multi Maze")
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

