import numpy as np
import pickle
import datetime
'''Implementation of standard version of the populat Quality Diversity algorithm MAP-Elites 
Pseudocode  and further information available: https://arxiv.org/pdf/1504.04909.pdf

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
    def __init__(self, individualSize, evalFunction, binCount = 5, maxVelocity = 0.125, maxAngularVelocity = 1.4):
        self.bins = binCount
        self.evalFunction = evalFunction
        #max observed velocity -.125 m/s
        self.velocity = [round(e * (maxVelocity/self.bins), 3) for e in range(1, (self.bins + 1))]
        #max observed angular velocity 2.1 rad/s
        self.angularV = [round(e * (maxAngularVelocity/self.bins), 3) for e in range(1, (self.bins + 1))]
        self.memberSize = individualSize
        self.mutationRate = 0.1
        self.tournamentSize = 10
        self.searchSpace = np.zeros([self.bins,self.bins])
        self.savedMembers = np.zeros([self.bins,self.bins, self.memberSize])
        self.saveFile = "EMAP-EasyRace 1.txt"
    
    def insert(self, behaviours, fitness, individual):
        c1 = np.digitize(behaviours[0], self.velocity)
        c2 = np.digitize(behaviours[1], self.angularV)
        if self.searchSpace[c1][c2] < fitness:
            self.searchSpace[c1][c2] = fitness
            self.savedMembers[c1][c2][:] = individual
    
    
    def newMember(self):
        newMember = np.zeros(self.memberSize)
        for i in range(self.memberSize):
            newMember[i] = np.random.uniform(-1, 1)
        
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
    
    def getRandomMember(self):
        c1 = np.random.randint(0, self.bins)
        c2 = np.random.randint(0, self.bins)
        while np.max(self.savedMembers[c1][c2]) == 0 or np.min(self.savedMembers[c1][c2]) == 0 :
            c1 = np.random.randint(0, self.bins)
            c2 = np.random.randint(0, self.bins)
        return self.savedMembers[c1][c2]
    
    def getRandomID(self):
        c1 = np.random.randint(0, self.bins)
        c2 = np.random.randint(0, self.bins)
        while np.max(self.savedMembers[c1][c2]) == 0:
            c1 = np.random.randint(0, self.bins)
            c2 = np.random.randint(0, self.bins)
        return [c1, c2]

    def saveMap(self):
#         x = datetime.datetime.now()
#         USE THIS ONE
#         date_time = x.strftime("%m-%d-%Y, %H-%M-%S")

#         date_time = date_time + ".txt"
#         date_time = "MAP - MiddleWall -  " + date_time
        f = open(self.saveFile,'wb')
        pickle.dump(zip(self.searchSpace, self.savedMembers), f)
        f.close()
        self.saveLogData()
        
    
    def saveLogData(self):
        dt = datetime.datetime.now()
        date_time = dt.strftime("%m/%d/%Y, %H:%M:%S")
        infoString = "Following data added on: " + date_time
        infoString += "\nGenerations: " + str(self.generations)
        infoString += "\nInitial Pop Size: " + str(self.initialPopSize)
        infoString += "\nMutationRate: " + str(self.mutationRate)
        infoString += "\n____________________________________________________"
        tf = open(self.saveFile + " DataLog.txt", 'w')
        tf.write(infoString)
        tf.close()
    
    def loadMap(self, saveFile):
        f = open(self.saveFile,'rb')
        data = list(zip(*(pickle.load(f))))
        for i in range(self.bins):
            for j in range(self.bins):
                self.searchSpace[i][j] = data[0][i][j]
                self.savedMembers[i][j] = data[1][i][j]
        self.loadedState = True

    def runAlgorithm(self, generations):
        print("Running MAP-Elites evaluation for " + str(generations) + " generations")
        
        self.initialPopSize = round(generations/10)
        
        if self.loadedState:
            self.initialPopSize = 0
            
        self.generations = generations
        gen = 0
        while gen < generations:
            newMember = np.zeros(self.memberSize)
            if gen < self.initialPopSize:
                newMember = self.newMember()
            else:
                newMember[:] = self.getRandomMember()
                newMember[:] = self.gaussianMutation(newMember)
            
            averageV, averageAV, fitness = self.evalFunction(newMember, True)
    
            behaviours = [averageV, averageAV]
            self.insert(behaviours, fitness, newMember)
            print(gen)
            gen += 1
    
        self.visualiseMap()

        self.saveMap()
        
    def visualiseMap(self):
        solutionCount = 0
        solutionSpace = np.zeros([self.bins,self.bins])
        for i in range(self.bins):
            for j in range(self.bins):
                if self.searchSpace[i][j] > self.solutionThreshold:
                    solutionCount += 1
                    solutionSpace[i][j] = 1
    
        print("total solutions found: ")
        print(solutionCount)
        print("solution space:")
        print(solutionSpace)
        print(self.searchSpace)
