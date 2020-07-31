from controller import Supervisor
from FixedNeuralNetwork import FixedNeuralNetwork
from RecurrentNeuralNetwork import RecurrentNeuralNetwork
from CMAES import CMAES, NCMAES
from MAPElites import MAPElites
import numpy as np
from DataReporting import DataReporting

'''
Universal supervisor controller for Webots Reinforcement Learning experimentation
              
QUICK START
To successfully initialise:
 - First, via the robot window or the Webots documentation, find the names of your robots distance sensors and place them into a string array like so: distanceSensors = ['cs0', 'cs1', 'cs2', 'cs3']
 - Now initialise desired type of Neural Network (fixed or recurrent) by initialising a new instance of desired class - i.e network = FixedNeuralNetwork(len(distanceSensors))
 - Create a new controller instance with desired values including passing the NN (robot and target names must be defined manually beforehand in DEF fields) - myController = UniversalController(network = network)
 - Pass your distance sensor array to your controller intance via the setDistanceSensors() function
 - Create instance of desired evolutionary strategy type (standard CMAES or novelty NCMAES) and pass desired solution size (taken from NN), reference to evaluation function myEA, and desired population size like so: CMAES(individualSize = network.solutionSize, evalFunction = myController.evaluateRobot, popSize = 20)
 - Call the runAlgorithm() method on your strategy instance
 
 - OPTIONAL - set custom space dimensions of your maze directly or with setDimensions function (default is a 1m x 1m search space)
 
 MAP ELITES - REMIND them that they need to know what maxV and maxAV they can reach with their chosen robot
 
 DATA REPORTING GENERAL
  - Make new instance of DataReporting class and pass in the desired save file name (otherwise will be saved as "undefined.txt")
  - Then pass your DataReporting instance to your controller instance to store as a variable like: myController.dataReporting = myData 
  - FOR SIGMA DataVisualisationandTesting you can now call the sigmaTest(self, myEA, upperLimit = 1.0, lowerLimit = 0.1) function from your controller instance with desired range
  - SIGMA DataVisualisationandTesting is only compatible/necessary with an instance of the CMAES or NCMAES class and will use the popSize already specified in your EA instance
  - SIGMA DataVisualisationandTesting is set to run 10 times for each sigma increment for 5 total generations based on pre set class variables in the DataReporting class - this can be changed if desired
  - Results will automatically be saved in a text file via pickle and input file names will automatically be appended with current date time - e.e "input 06-29-2020, 19-29-58.txt"
 DATA REPORTING ALGORITHM RUN
  - Once set up as per first two steps above, you can now call the algorithmTest function like so: myController.algorithmTest(myEA, generations = 5, totalRuns = 50)
  - These results will also be saved in a text file via pickle
 
 DATA REPORTING LOADING AND DISPLAYING DATA
  - Data can now be loaded and displayed for either sigma or algorithm DataVisualisationandTesting in a box plot using whatever IDE you wish
  - Simply create a new DataRerpoting class instance, passing the name of the file you wish to load like so: myData = DataReporting("Sigma DataVisualisationandTesting - MiddleWall -  06-29-2020, 19-29-58.txt")
  - IF LOADING SIGMA DATA - first specify the lower and upper limits used like so: myData.upperLimit = 1.0
                                                                                   myData.lowerLimit = 0.1
  - Now specify the maxEvals that were granted to each run (default is set to 100 in-class)
  - Now call the myData.loadData() function
  - To display box plots now call either the myData.displaySigmaBoxPlots() for SIGMA DataVisualisationandTesting results, or myData.displayAlgorithmBoxPlots("CMA-ES") passing in the string/s you wish to be represented on the X axis
    EXAMPLE CODE FOR LOADING AND DISPLAYING SIGMA RESULTS
    myData = DataReporting("Sigma DataVisualisationandTesting - MiddleWall - 100ME, 0.1 - 1.0, 06-29-2020, 19-29-58.txt")
    myData.upperLimit = 1.0
    myData.lowerLimit = 0.1
    myData.maxEvals = 100
    myData.loadData()
    myData.displaySigmaBoxPlots()
  
  - ADVANCED - multiple load files can be combined by simply calling the loadData function and passing in the name of the file you wish to load more data from. This can be used to compare multiple algorithms or experiments in one box plot automatically, like so:
    EXAMPLE CODE DISPLAYING BOX PLOTS FOR TWO RUNS OF CMAES
    myData = DataReporting("Sigma DataVisualisationandTesting - MiddleWall -  06-29-2020, 19-29-58.txt")
    myData.maxEvals = 100
    myData.loadData()
    myData.loadData("CMA-ES Test-MiddleWall-FNN07-16-2020, 16-52-07.txt")
    myData.displayAlgorithmBoxPlots("CMA-ES", "CMA-ES")
    myData.displayBoxPlots()
    

  
'''
class UniversalController:
    def __init__(self, network, robotName = "standard", targetName = "TARGET", numOfInputs = 4, evalRuntime = 90, timeStep = 32):
        self.neuralNetwork = network
        self.TIME_STEP = timeStep #standard time step is 32 or 64
        self.supervisor = Supervisor()
        self.inputs = numOfInputs
        self.solutionThreshold = 0.95
        #call methods functions to intialise robot and target
        self.getAndSetRobot(robotName)
        self.getAndSetTarget(targetName)
        self.evalRuntime = evalRuntime
        
        #default to 1m x 1m space but can be edited directly or using method below
        self.maxDistance = 1.4142
        
        self.dataReporting = DataReporting()
       
    def getAndSetRobot(self, robotName):
        # initialise robot node and components
        self.robotNode = self.supervisor.getFromDef(robotName)#TARGET ROBOT MUST BE NAMED IN DEF FIELD
        self.robotTrans = self.robotNode.getField("translation")
        self.robotRotation = self.robotNode.getField("rotation")
        #call start location function to store initial position
        self.getAndSetRobotStart()
        #call motors function
        self.getAndSetMotors()
        
    def enableDataReporting(self, saveFileName):
        self.dataReporting.saveFileName = saveFileName
        
    def loadData(self, loadFileName):
        self.myData.loadFileName = loadFileName
        
    def getAndSetRobotStart(self):
        #establish robot starting position
        self.robotStartingLocation = self.robotTrans.getSFVec3f()
        self.robotStartingRotation = self.robotRotation.getSFRotation()
    
    '''EDIT HERE TO EXPAND TO DIFFERENT NUMBERS OF WHEELS/MOTORS'''
    def getAndSetMotors(self):
        self.motors = []
        self.motorMaxVs = []
        #get robot motors - currently works for all two wheel motor morphologies
        self.leftMotor = self.supervisor.getMotor('left wheel motor')
        self.rightMotor = self.supervisor.getMotor('right wheel motor')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        
        #get the max velocity each motor is capable of and set the max velocity
        self.leftMotorMax = self.leftMotor.getMaxVelocity()
        self.rightMotorMax = self.rightMotor.getMaxVelocity()
        
        #append necessary additional motors and max velocities here after enabling as above
        self.motors.append(self.leftMotor)
        self.motors.append(self.rightMotor)
        self.motorMaxVs.append(self.leftMotorMax)
        self.motorMaxVs.append(self.rightMotorMax)
    
    def getAndSetTarget(self, targetName):
        #get target location
        self.target = self.supervisor.getFromDef(targetName)#TARGET MUST BE NAMED IN DEF FIELD
        #get and set the translation and location for retrieval when needed
        self.targetTrans = self.target.getField("translation")
        self.targetLocation = self.targetTrans.getSFVec3f()
        
    def setDimensions(self, spaceDimensions):
        #basic pythagorean calculation to find max distance possible in square space
        self.maxDistance = np.sqrt(spaceDimensions[0]**2 + spaceDimensions[1]**2)
        #print(self.maxDistance)
    
    def setDistanceSensors(self, distanceSensors):
        #takes in array of strings - names of distance sensors
        self.distanceSensors = []
        self.minDSvalues = []
        self.DSValueRange = []
        for i in range(len(distanceSensors)):
            #set and enable each sensor
            self.distanceSensors.append(self.supervisor.getDistanceSensor(distanceSensors[i]))
            self.distanceSensors[i].enable(self.TIME_STEP)
            #get and store the min reading value of each sensor
            self.minDSvalues.append(self.distanceSensors[i].getMinValue())
            #get and store the possible value range of each sensor
            self.DSValueRange.append(self.distanceSensors[i].getMaxValue() - self.minDSvalues[i])
        
        #print(self.DSValueRange)
    
    
    def getDSValues(self):
        #get distance sensor values
        values = []
        for i in range(len(self.distanceSensors)):
            value = self.distanceSensors[i].getValue()
            #value = value/self.maxDSReading #practical max value
            value = value - self.minDSvalues[i]
            if value < 0.0:
                value = 0.0 #to account for gaussian noise
            value = value/(self.DSValueRange[i])
            #account for gaussian noise providing higher than max reading
            if value > 1.0:
                value = 1.0
            values.append(value)
        #return a list of the normalised sensor readings
        return values

    def computeMotors(self, DSValues):
        #get the outputs of the neural network and convert into wheel motor speeds
        #already fully flexible for multiple motors
        results = self.neuralNetwork.forwardPass(DSValues) 
        for i in range(len(self.motors)):
            self.motors[i].setVelocity(results[i]*self.motorMaxVs[i])

    
    def resetAllPhysics(self):
        #reset robot physics and return to starting translation ready for next run
        self.robotRotation.setSFRotation(self.robotStartingRotation)
        self.robotTrans.setSFVec3f(self.robotStartingLocation)
        self.robotNode.resetPhysics()
    
    '''SIMULATION FUNCTION - can also be used directly for manual DataVisualisationandTesting of weight arrays (to manually repeat successful solutions etc.)'''
    def evaluateRobot(self, individual, mapElites = False):
        self.neuralNetwork.decodeEA(individual) #individual passed from cma
        t = self.supervisor.getTime()
        
        if mapElites:
            velocity = []
            angularV = []   
            
        while self.supervisor.getTime() - t < self.evalRuntime:
            
            self.computeMotors(self.getDSValues())
            currentFit = self.calculateFitness()
            
            if mapElites:
                currentV = self.robotNode.getVelocity()
                velocity.append(np.sqrt((currentV[0]**2) + (currentV[1]**2) + (currentV[2]**2)))
                angularV.append(np.sqrt(currentV[4]**2))
            
            '''MAYBE CHANGE TO CLASS VARIABLE'''
            if currentFit > self.solutionThreshold:
                timeTaken = self.supervisor.getTime() - t
                #safety measure due to timestep and thread lag
                if timeTaken > 0.0:  #then a legitimate solution
                    fit = self.calculateFitness()
                    break
            if self.supervisor.step(self.TIME_STEP) == -1:
                quit()
        
        endpoint = self.robotTrans.getSFVec3f()[0: 3: 2]
        self.resetAllPhysics()
        #find fitness
        fit = self.calculateFitness()
        if mapElites:
            averageVelocity = np.average(velocity)
            averageAngularV = np.average(angularV)
            return averageVelocity, averageAngularV, fit
        return fit, endpoint
        
    def calculateFitness(self):
        values = self.robotTrans.getSFVec3f()
        distanceFromTarget = np.sqrt((self.targetLocation[0] - values[0])**2 + (self.targetLocation[2] - values[2])**2)
        fit = 1.0 - (distanceFromTarget/ self.maxDistance)
        return fit
    
    def setStrategy(self, strategy):
        self.strategy = strategy
    
    def sigmaTest(self, myEA, upperLimit = 1.0, lowerLimit = 0.1):
        self.dataReporting.sigmaTest(myEA, upperLimit, lowerLimit)
        
    def algorithmTest(self, myEA, generations, totalRuns):
        self.dataReporting.algorithmTest(myEA, generations, totalRuns)

def main():
    '''STEP 1: Create an array for your robot's distance sensors (names can be found in documentation or robot window)'''
    distanceSensors = ['cs0', 'cs1', 'cs2', 'cs3']
    #Another example distance sensor array:
    #distanceSensors = ['ds0', 'ds1', 'ds2', 'ds3', 'ds4', 'ds5', 'ds6', 'ds7']
    '''STEP 2: Create your Neural Network instance '''
    network = FixedNeuralNetwork(inputs = len(distanceSensors))
    #network = RecurrentNeuralNetwork(inputs = len(distanceSensors))
    '''STEP 3: Create your controller instance (pass in the network) '''
    myController = UniversalController(network = network)
    #optional - default is set to 90 seconds
    myController.evalRuntime = 100

    '''STEP 4: Pass your distance sensor array to your controller'''
    myController.setDistanceSensors(distanceSensors)
    #optional - set size of your environment, default is 1m x 1m
    #myController.setDimensions([0.75, 0.75])
    
    '''STEP 5: Create your algorithm instance '''
    #myEA = CMAES(individualSize = network.solutionSize, evalFunction = myController.evaluateRobot, popSize = 50, sigma=1.0)
    #myEA = NCMAES(individualSize = network.solutionSize, evalFunction = myController.evaluateRobot, popSize = 50)
    myEA = MAPElites(network.solutionSize, myController.evaluateRobot)
    #OPTIONAL FOR MAP ELITES - LOAD SAVED MAP
    #myEA.loadMap(myEA.saveFile)
    
    '''STEP 6: You can now run a test with your chosen set up'''
    #myEA.runAlgorithm(generations = 40)
    
    ''' ****************************************************************** '''
    '''OPTIONAL Data reporting 1: Create an instance of the DataReporting class with your desired filename '''
    myData = DataReporting("H-CMAES FULL TEST - EscapeRoom, 1.0N, FNN, ")
    
    '''OPTIONAL Data reporting 2: Pass your DataReporting instance to your controller '''
    myController.dataReporting = myData
    #myEA.runAlgorithm(generations = 200)
    '''OPTIONAL Data reporting 3: You can now run a recorded algorithm test '''
    #myController.algorithmTest(myEA, generations = 200, totalRuns = 10)
    myEA.saveFile = "EasyRace initial test"
    myEA.runAlgorithm(2500)

    ''' ****************************************************************** '''
    '''OPTIONAL Sigma DataVisualisationandTesting (compatible with CMA-ES and NCMA-ES: You can also run Sigma testing with your chosen set up'''
    #myData.sigmaGenerations = 6
    #myData.sigmaRuns = 25
    #myController.sigmaTest(myEA, upperLimit=2.0, lowerLimit= 0.1)
    
    ''' ****************************************************************** '''
    '''OPTIONAL MANUAL INDIVIDUAL TESTING: You can also manually test an array of individual NN weights outside of any algorithm'''
    #individual solution for easy race with e-puck 1mx1m Solution found! On eval: 5607
    #individual = [0.8097185327742493, 1.0, 1.0, 1.0, -1.0, 1.0, 0.6659693031947151, 1.0, -0.7128410988986517, 0.7759888379310794, 0.6158819263907493, -0.5002185658710185, -0.4114915190302556, 0.3945365591958676, -1.0, -1.0, -0.5357976419238993, 0.686794799926385, 0.37363019778569717, -0.8632204600054812, 0.04448374706105757, -1.0, -0.14329702665397612, 1.0, -0.19933385055678499, 1.0, -0.03975008945789771, 0.5962539214037185, 0.27705248801127647, 0.7863774637788581, 0.6652600051159849, -0.01746922821459823, -0.36609418280610356, 0.8388881260495615, -0.9864790228200031, 0.21640543195173925, -0.9052129595083868, 0.6892293928669115, 0.19209996575514293, 0.12432272462382862, 0.4424098393060318, 1.0, 0.1993979077361407, 1.0, -0.4578395528108651, 0.7647578741175924, -0.5486539941695451, 0.39743749941275147, 0.1595009727468328, 0.5908481373295096, -0.4237835873171585, -0.840883947320976, -0.29148155790043917, 0.0857396929554975, 1.0, 0.08364818537527521, -0.14681859725698373, 1.0, -1.0, 0.9052605393152322, 0.4701886718422032, 1.0, 0.3166733018899936, 1.0, 0.3117077966988857, 1.0, 0.2641816170544849, -1.0, -0.19602972568359187, 0.8962542201946909, -1.0, 1.0, -0.28612124049645205, 0.18711138433588154, -1.0, -0.5591113644985205, -0.03062881835647905, -0.6285350575016602, -1.0, -1.0, 0.9487048419607723, -0.977797554211353, -0.5603219629469457, 0.7475512009902726, -0.7663071478431556, -1.0, 0.40344970445360867, 0.3699638906548576, 0.7268674163984049, 0.07247054687042216, -0.33889224607397994, -0.37444439067419233, -0.002868698310303198, 1.0, -0.009536072434543618, 1.0, 0.33814644785550724, 0.10667125415244208, 0.33107155369435026, -1.0, 0.043189588905676195, -0.7291550894602776, 0.973776656547831, -0.8012657905755017, 0.4071690477578325, 1.0, 0.6324462154841985, -0.7737891443111629, 0.13118806189133975, -0.3461497228646832, -0.3252008724006586, -1.0, -0.1515837402278993, -0.5424979069856092, 1.0, 0.6371076012849336, -1.0, 0.023976441245781906, 0.77068578760371, 0.7291333558530215, -0.9194146138934837, 1.0, 1.0, 1.0, -0.14040864516006357, 0.41416268073482193, 1.0, 0.4669196459489035, 1.0, 0.40772827297083725, 0.3986990273112243, -0.9608307977197296, 0.9763764300951379, -0.44879592053469497, -0.8391178182959773, -0.12609468450988576, -0.00853832005574566, -0.22646607877693642, -1.0, -0.9386439716407828, -0.5312671127172448, 1.0, -1.0, 0.09280516180767581, -0.8121358169332663, -0.5425053685622886, 1.0, 1.0, 0.5804424069295344, -0.8470867989060451, 1.0, 0.12588224952776308, 0.40064596955065274, -0.9515528994564351, -1.0, 0.14437987536310232, 1.0, 1.0, 1.0, -0.3999986542362194, -0.5669744633504175, -0.6133345970702229, -0.5984395950654815, 1.0, 0.008264524132969346, -1.0, -0.3471302679505146, -0.7592449623307239, -1.0, 1.0, 1.0, 1.0, 0.37877000088676116, 0.9433205124543965, -0.7740743104681928, -0.07446994213057241, 0.8392254737949248, -0.2406409162230742]    
    #fit = myController.evaluateRobot(individual)

main()
