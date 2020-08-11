import numpy as np
import pickle
import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

'''Implementation of customised version of the popular Quality Diversity algorithm MAP-Elites 
Pseudocode  and further information available: https://arxiv.org/pdf/1504.04909.pdf

Customisation:
Tournament selection is used to select cell of next individual to mutate, prioritising cells with the most empty adjascent cells

MAP-Elites takes, in our case, 2 different behavioural descriptors and creates a 2D map of cells that each hold a record of the 
highest performing member of that part in the behavioural space.

In short, this is to prioritise greater diversity of solutions in the understanding that this diversity will eventually lead to solutions for deceptive tasks
that other algorithms are failing to find.

This implementation is set to a relatively small number of bins or 'cells' and uses average velocity and average angular velocity as behavioural descriptors

At the end of each run, a datetime stamped map is saved with the saveFile variable as a name which can be loaded and used in future runs
A separate file will also be saved/updated with a data log detailing the run details and parameters to help keep track of the maps development
'''


class MAPElites():
    np.random.seed()
    solutionThreshold = 0.95
    loadedState = False
    '''The below maxVeloity and maxAngularVelocity applies to the e-puck robot and will need to be manually changed for different morphologies'''
    '''Limitation of Webots - can't directly access what the maximum angular velocity is for a given robot - has to be determined manually through testing in simulation '''
    
    #maxDFS in easyrace = 1.23138
    #maxdfs in multimaze = 0.78749
    #maxdfs in middlewall = 0.8942810
    #maxdfs in escape room 0.63008 (already accounted for)
    
    heatmapTitle = "undefined"
    yAxisLabel = "Average Velocity (m/s)"
    
    def __init__(self, individualSize, evalFunction, binCount = 10, tournamentSize = 5, maxVelocity = 0.125, maxDistance = 1.2):
        self.bins = binCount
        self.evalFunction = evalFunction
        #max observed velocity -.125 m/s
        self.velocity = [round(e * (maxVelocity/self.bins), 3) for e in range(1, (self.bins + 1))]
        #max observed angular velocity 2.1 rad/s
        self.distanceFromStart = [round(e * (maxDistance/self.bins), 3) for e in range(1, (self.bins + 1))]
        self.memberSize = individualSize
        self.mutationRate = 0.1
        self.tournamentSize = tournamentSize
        self.searchSpace = np.zeros([self.bins,self.bins])
        self.savedMembers = np.zeros([self.bins,self.bins, self.memberSize])
        #saves any successful solutions in format: [individual, coordinates, eval]
        self.successes = []
        self.saveFile = "EMAP - Unspecified"
    
    def insert(self, behaviours, fitness, individual):
        c1 = np.digitize(behaviours[0], self.velocity)
        c2 = np.digitize(behaviours[1], self.distanceFromStart)
        
        #finds the cell the individual belongs to, keeps if better fitness
        if self.searchSpace[c1][c2] < fitness:
            self.searchSpace[c1][c2] = fitness
            self.savedMembers[c1][c2][:] = individual
    
        return [c1, c2]
    def newMember(self):
#         newMember = np.zeros(self.memberSize)
#         for i in range(self.memberSize):
#             newMember[i] = np.random.uniform(-1, 1)
        newMember = np.random.uniform(-1, 1, self.memberSize)
        return newMember
    
    def gaussianMutation(self, individual):
        #create boolean array determining whether to mutate a given index or not
        mutate = np.random.uniform(0, 1, individual.shape) < self.mutationRate
   
        for i in range(len(individual)):
            if mutate[i]:
                individual[i] = np.random.normal(0, 0.5)
                if individual[i] > 1.0:
                    individual[i] = 1.0
                if individual[i] < -1.0:
                    individual[i] = -1.0
    
        return individual
    
    def checkEmptyAdjascent(self, member):
        #check number of empty bins next to member
        emptyCount = 0
        for i in range(member[0]-1, member[0]+2):
            for j in range(member[1]-1, member[1]+2):
                #ensure we are not checking member or index outside of searchspace
                if [i, j] == member or i < 0 or j < 0 or i >= self.bins or j >= self.bins:
                    pass
                else:
                    if self.searchSpace[i][j] == 0:
                        emptyCount += 1
                    else:
                        pass
        return emptyCount
        
    def combineLoadedMap(self, filename):
        #for use when plotting combined heatmap from multiple run results
        f = open(filename,'rb')
        data = list(zip(*(pickle.load(f))))
        for i in range(self.bins):
            for j in range(self.bins):
                if data[0][i][j] > self.searchSpace[i][j]:
                    self.searchSpace[i][j] = data[0][i][j]
                    
    def getRandomMember(self):
        c1 = np.random.randint(0, self.bins)
        c2 = np.random.randint(0, self.bins)
        while np.max(self.savedMembers[c1][c2]) == 0 or np.min(self.savedMembers[c1][c2]) == 0 :
            c1 = np.random.randint(0, self.bins)
            c2 = np.random.randint(0, self.bins)
        
        return self.savedMembers[c1][c2]
    
    def tournamentSelect(self):
        #select first initial random choice and set to best choice so far
        bestMemberID = self.getRandomID()
        mostEmptyBins = self.checkEmptyAdjascent(bestMemberID)
        
        for round in range(self.tournamentSize-1):
            #select new random member
            tempMemberID = self.getRandomID()
            tempEmptyBins = self.checkEmptyAdjascent(tempMemberID)
            #if new choice has more empty nearby cells, set as best choice
            if tempEmptyBins > mostEmptyBins:
                mostEmptyBins = tempEmptyBins
                bestMemberID = tempMemberID
        #retrieve the chosen member itself from the saved members array
        return self.savedMembers[bestMemberID[0]][bestMemberID[1]]
    def getRandomID(self):
        c1 = np.random.randint(0, self.bins)
        c2 = np.random.randint(0, self.bins)
        while np.max(self.searchSpace[c1][c2]) == 0:
            c1 = np.random.randint(0, self.bins)
            c2 = np.random.randint(0, self.bins)
        return [c1, c2]

    def saveMap(self):
        rawDT = datetime.datetime.now()
        date_time = rawDT.strftime("%m-%d-%Y, %H-%M-%S") #make a filename compatible datetime string
        parameters = " " + str(self.tournamentSize) + "TS, " + str(self.bins) + "BC, " + str(self.generations) + "gens" 
        self.saveFile = self.saveFile + parameters + date_time
        f = open(self.saveFile + ".txt",'wb')
        pickle.dump(zip(self.searchSpace, self.savedMembers), f)
        f.close()
        self.saveLogData()
#         if len(self.successes) > 0:
#             self.saveSuccesses()
            
    def saveSuccesses(self):
        f = open(self.saveFile + " SUCCESSES.txt",'wb')
        pickle.dump(self.successes, f)
        f.close()
        
    def saveLogData(self):
        dt = datetime.datetime.now()
        date_time = dt.strftime("%m/%d/%Y, %H:%M:%S")
        infoString = "Following data added on: " + date_time
        infoString += "\nBin count: " + str(self.bins)
        infoString += "\nGenerations: " + str(self.generations)
        infoString += "\nInitial Pop Size: " + str(self.initialPopSize)
        infoString += "\nMutationRate: " + str(self.mutationRate)
        infoString += "\nTournamentSize: " + str(self.tournamentSize)
        infoString += "\nSuccesses found: " + str(len(self.successes))
        infoString += "\nSuccessful cells: " + str(self.solutionCount)
        if len(self.successes) > 0:
            infoString += "\nFirst successful solution found on eval: " + str(self.successes[0][2])
        infoString += "\n____________________________________________________"
        tf = open(self.saveFile + " DataLog.txt", 'w')
        tf.write(infoString)
        tf.close()
    
    def loadMap(self, saveFile):
        f = open(saveFile,'rb')
        data = list(zip(*(pickle.load(f))))
        for i in range(self.bins):
            for j in range(self.bins):
                self.searchSpace[i][j] = data[0][i][j]
                self.savedMembers[i][j] = data[1][i][j]
        self.loadedState = True

    
    def refresh(self):
        self.searchSpace[:] = np.zeros([self.bins,self.bins])
        self.savedMembers[:] = np.zeros([self.bins,self.bins, self.memberSize])
        self.successes[:] = []
    #minimum effective generations is 10
    def runAlgorithm(self, generations):
        
        np.random.seed()
        print("Running MAP-Elites evaluation for " + str(generations) + " generations with a bin count of: " + str(self.bins))
        
        #use the first 10% of the total generations to generate initial cell entries or bincount *50 - whatever smaller
        self.initialPopSize  = round(generations/10)
        
        #check if map is already loaded
        if self.loadedState:
            self.initialPopSize = 0
        else:
            #if not its a new run so refresh class vairables
            self.refresh()
            
        self.generations = generations
        gen = 0
        while gen < generations:
            #create a new empty array of correct size
            newMember = np.zeros(self.memberSize)
            #if we are still running the initial batch, randomly create a new member
            if gen < self.initialPopSize:
                newMember[:] = self.newMember()
            else:
                #otherwise use tournament select to chose a cell to mutate
                newMember[:] = self.tournamentSelect()
                newMember[:] = self.gaussianMutation(newMember)
            
            #get the behavioural description of the new member through evaluation
            averageV, distanceFS, fitness = self.evalFunction(newMember, True)
            
            behaviours = [averageV, distanceFS]
            ##pass to insert function which determines whether to keep or discard
            coordinates = self.insert(behaviours, fitness, newMember)
            
            #check if it's a successful solution
            if fitness > self.solutionThreshold:
                self.successes.append([newMember, coordinates, gen])
                print("Solution found! On eval: " + str(gen) + ", at map coordinate: " + str(coordinates[0]) + "," + str(coordinates[1]))
                break
            #every 100 iterations, prints the searchspace to the console for monitoring
            if gen % 100 == 0 and gen > 0:
                print(gen)
                print(self.searchSpace)
            gen += 1
        
        #visualise the map and the solutions found in the console
        self.visualiseMap()
        self.saveMap()
        return gen
        
        
    def visualiseMap(self):
        self.solutionCount = 0
        solutions = []
        solutionSpace = np.zeros([self.bins,self.bins])
        for i in range(self.bins):
            for j in range(self.bins):
                if self.searchSpace[i][j] > self.solutionThreshold:
                    self.solutionCount += 1
                    solutionSpace[i][j] = 1
                    solutions.append(self.savedMembers[i][j])
        
        print("total solutions found: ")
        print(self.solutionCount)
        print("solution space:")
        print(solutionSpace)
        print(self.searchSpace)
        return solutions
    
    def generateHeatmap(self):
        df = pd.DataFrame(self.searchSpace)
        ax = sns.heatmap(df)
        ax.set_xticklabels(self.distanceFromStart,
                                    rotation=0, fontsize=12)
        ax.set_yticklabels(self.velocity,rotation=0, fontsize=12)
        ax.set_xlabel("Distance from Start (m)", fontsize=14)
        ax.set_ylabel("Average Velocity (m/s)", fontsize=14)
        ax.set_title("MAP-Elites Heatmap - Easy Race - Custom e-puck - RNN", fontsize=14)
        plt.show()
