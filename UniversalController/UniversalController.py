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
 - create instance of desired evolutionary strategy type (standard CMAES or novelty NCMAES) and pass desired solution size (taken from NN), reference to evaluation function myEA, and desired population size like so: CMAES(individualSize = network.solutionSize, evalFunction = myController.evaluateRobot, popSize = 20)
 - call the runAlgorithm() method on your strategy instance
 
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
        self.targetTrans = self.target.getField("translation")
        self.targetLocation = self.targetTrans.getSFVec3f()
        
    def setDimensions(self, spaceDimensions):
        #basic pythagorean calculation to find max distance possible in square space
        self.maxDistance = np.sqrt(spaceDimensions[0]**2 + spaceDimensions[1]**2)
        print(self.maxDistance)
    
    def setDistanceSensors(self, distanceSensors):
        #takes in array of strings - names of distance sensors and int of max possible reading
        #optional minimum base reading if value above 0 - default is set to 0
        self.distanceSensors = []
        self.minDSvalues = []
        self.DSValueRange = []
        for i in range(len(distanceSensors)):
            #set and enable each sensor
            self.distanceSensors.append(self.supervisor.getDistanceSensor(distanceSensors[i]))
            self.distanceSensors[i].enable(self.TIME_STEP)
            self.minDSvalues.append(self.distanceSensors[i].getMinValue())
            self.DSValueRange.append(self.distanceSensors[i].getMaxValue() - self.minDSvalues[i])
        
        print(self.DSValueRange)
    
    #get distance sensor values
    def getDSValues(self):
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
    '''STEP 1: Create an array for your robot's distance sensors '''
    #distanceSensors = ['cs0', 'cs1', 'cs2', 'cs3']
    #Another example distance sensor array:
    distanceSensors = ['ds0', 'ds1', 'ds2', 'ds3', 'ds4', 'ds5', 'ds6', 'ds7']
    '''STEP 2: Create your Neural Network instance '''
    #network = FixedNeuralNetwork(inputs = len(distanceSensors))
    network = RecurrentNeuralNetwork(inputs = len(distanceSensors))
    '''STEP 3: Create your controller instance (pass in the network) '''
    myController = UniversalController(network = network)
    #optional - default is set to 90 seconds
    myController.evalRuntime = 100

    '''STEP 4: Pass your distance sensor array to your controller - with max and minimum sensor range (can be found in documentation or robot window) '''
    myController.setDistanceSensors(distanceSensors)
    #optional - set size of your environment, default is 1m x 1m
    #myController.setDimensions([0.75, 0.75])
    
    '''STEP 5: Create your algorithm instance '''
    myEA = NCMAES(individualSize = network.solutionSize, evalFunction = myController.evaluateRobot, popSize = 50, sigma=1.0)
    #myEA = NCMAES(individualSize = network.solutionSize, evalFunction = myController.evaluateRobot, popSize = 50)
    #myEA = MAPElites(network.solutionSize, myController.evaluateRobot)
    #OPTIONAL FOR MAP ELITES - LOAD SAVED MAP
    #myEA.loadMap(myEA.saveFile)
    
    '''STEP 6: You can now run a test with your chosen set up'''
    #myEA.runAlgorithm(generations = 40)
    
    ''' ****************************************************************** '''
    '''OPTIONAL Data reporting 1: Create an instance of the DataReporting class with your desired filename '''
    myData = DataReporting("HEMISSON - NCMAES FULL TEST - EasyRace, 1.0N, RNN,  ")
    
    '''OPTIONAL Data reporting 2: Pass your DataReporting instance to your controller '''
    myController.dataReporting = myData
    #myEA.runAlgorithm(generations = 200)
    '''OPTIONAL Data reporting 3: You can now run a recorded algorithm test '''
    myController.algorithmTest(myEA, generations = 200, totalRuns = 10)

    ''' ****************************************************************** '''
    '''OPTIONAL Sigma DataVisualisationandTesting (compatible with CMA-ES and NCMA-ES: You can also run Sigma testing with your chosen set up'''
    #myData.sigmaGenerations = 6
    #myData.sigmaRuns = 25
    #myController.sigmaTest(myEA, upperLimit=2.0, lowerLimit= 0.1)
    
    ''' ****************************************************************** '''
    '''OPTIONAL MANUAL INDIVIDUAL TESTING: You can also manually test an array of individual NN weights outside of any algorithm'''
    #individual solution for easy race with e-puck 1mx1m Solution found! On eval: 5607
    #individual = [0.31031731300992854, -0.625717838677998, 0.6822099211950758, 0.35093854856430307, 1.0, 1.0, 0.0009309922329320541, -0.6667059438165372, 1.0, 0.7686658787840771, 0.9829338368777658, -1.0, 1.0, -0.03535018183798579, 0.9159827029809522, 1.0, -1.0, 0.9261351148297444, 0.9370941269429579, 0.09611646569838542, -0.8663865162282224, 1.0, -0.028251179649589866, 0.4061503955849184, -0.08250667681148824, -0.16746124377877428, 0.8178337947911983, -1.0, 0.3535909339816691, -1.0, 1.0, 1.0, -0.5656150166799535, 1.0, -0.16126224128955827, 1.0, -1.0, -0.29056750820102073, -0.8854209318538603, -1.0, 0.24688775452274014, -0.12487889493973085, 0.29071055270549007, -0.7651515341806517, -1.0, -1.0, -1.0, -0.31465437442427496, 0.2510766867974456, 0.5723782486620624, 0.9915336601264425, 0.3116319685494229, -1.0, -1.0, 0.7363206547132071, 0.05777593477124992, 0.19188887789494083, -0.20578177616681684, -0.8668258863884903, -0.6021788695367846, 0.5183740889343345, -1.0, 0.5500037481213166, 1.0, -1.0, -1.0, -0.44165738850252706, 0.43049036826016873, 0.6546771865727868, 0.4446467180913747, 0.39350581662002176, 0.9305053003498948, -0.4935008176214979, -0.06245987808487186, 0.6857409432069611, -0.452700509893487, 1.0, -1.0, -0.06842486278572765, -0.4570739508402518, -0.7228030774278107, 0.7618130695729538, 0.9686366678296847, 0.6605559838799274, 0.9137264735414016, -1.0, -0.8135467077346951, 0.7215679264886387, -1.0, -0.5613849269551666, 0.2853374933108166, -1.0, 1.0, -1.0, 0.6405666793785363, -0.2875108805645757, 0.5121031497618143, -0.14452536128233334, -0.437908361380963, -0.33012555615715544, 1.0, -0.21844058248302184, 0.16582261332465467, -0.005051709103707819, -0.9606691348098384, 0.5535601518474995, -0.5266974330838976, 1.0, -0.04729877975728782, 1.0, 0.05560175660985446, 0.682228882537105, 0.5994424338777845, 0.7824572446153184, -0.007316007687599012, 0.32040004920778414, -1.0, 0.5046873109456637, -1.0, 0.12600552901966905, -1.0, 1.0]
    #fit = myController.evaluateRobot(individual)

main()
