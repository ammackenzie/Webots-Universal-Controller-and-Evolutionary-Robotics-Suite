from controller import Supervisor
from FixedNeuralNetwork import FixedNeuralNetwork
from RecurrentNeuralNetwork import RecurrentNeuralNetwork
from CMAES import CMAES, NCMAES
from MAPElites import MAPElites
import numpy as np
from DataReporting import DataReporting

'''
Universal supervisor controller for Webots Reinforcement Learning experimentation
Requirements: Deap Library: https://deap.readthedocs.io/en/master/installation.html
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
    
    def setDistanceSensors(self, distanceSensors, maxValue, minReading = 0):
        #takes in array of strings - names of distance sensors and int of max possible reading
        #optional minimum base reading if value above 0 - default is set to 0
        self.distanceSensors = []
        for i in range(len(distanceSensors)):
            #set and enable each sensor
            self.distanceSensors.append(self.supervisor.getDistanceSensor(distanceSensors[i]))
            self.distanceSensors[i].enable(self.TIME_STEP)
        
        self.maxDSReading = maxValue
        self.minDSReading = minReading
    
    #get distance sensor values
    def getDSValues(self):
        values = []
        for i in range(len(self.distanceSensors)):
            value = self.distanceSensors[i].getValue()
            value = value/self.maxDSReading #practical max value
            #account for gaussian noise providing higher than max reading
            if value > 1.0:
                value = 1.0
            elif self.minDSReading > 0:
                value = value - (self.minDSReading/self.maxDSReading) #equivalent of base min reading 
                #to account for gaussian noise providing less than min reading
                if value < 0.0:
                    value = 0.0
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
    distanceSensors = ['cs0', 'cs1', 'cs2', 'cs3']
    #Another example distance sensor array:
    #distanceSensors = ['ds0', 'ds1', 'ds2', 'ds3', 'ds4', 'ds5', 'ds6', 'ds7']
    '''STEP 2: Create your Neural Network instance '''
    network = FixedNeuralNetwork(inputs = len(distanceSensors))
    #network = RecurrentNeuralNetwork(inputs = len(distanceSensors))
    '''STEP 3: Create your controller instance (pass in the network) '''
    myController = UniversalController(network = network)
    #optional - default is set to 0 seconds
    myController.evalRuntime = 100

    '''STEP 4: Pass your distance sensor array to your controller - with max and minimum sensor range (can be found in documentation or robot window) '''
    myController.setDistanceSensors(distanceSensors, 1000, 0)
    #optional - set size of your environment, default is 1m x 1m
    #myController.setDimensions([0.75, 0.75])
    
    '''STEP 5: Create your algorithm instance '''
    myEA = CMAES(individualSize = network.solutionSize, evalFunction = myController.evaluateRobot, popSize = 50)
    #myEA = NCMAES(individualSize = network.solutionSize, evalFunction = myController.evaluateRobot, popSize = 50)
    #myEA = MAPElites(network.solutionSize, myController.evaluateRobot)
    #OPTIONAL FOR MAP ELITES - LOAD SAVED MAP
    #myEA.loadMap(myEA.saveFile)
    
    '''STEP 6: You can now run an algorithm test with your chosen set up'''
    #myEA.runAlgorithm(generations = 40)
    
    ''' ****************************************************************** '''
    '''OPTIONAL Data reporting 1: Create an instance of the DataReporting class with your desired filename '''
    myData = DataReporting("Sigma Test - MultiMaze")
    
    '''OPTIONAL Data reporting 2: Pass your DataReporting instance to your controller '''
    myController.dataReporting = myData
    
    '''OPTIONAL Data reporting 3: You can now run an algorithm test '''
    #myController.algorithmTest(myEA, generations = 5, totalRuns = 50)

    ''' ****************************************************************** '''
    '''OPTIONAL Sigma DataVisualisationandTesting (compatible with CMA-ES and NCMA-ES: You can also run Sigma testing with your chosen set up'''
    myData.sigmaGenerations = 6
    myData.sigmaRuns = 25
    myController.sigmaTest(myEA, upperLimit=2.0, lowerLimit= 0.1)
    
    ''' ****************************************************************** '''
    '''OPTIONAL MANUAL INDIVIDUAL TESTING: You can also manually test an array of individual NN weights outside of any algorithm'''
    #individual = [0.5216170885274278, 0.2895953015451246, 0.06382171554745152, -0.08998482231248986, 0.2639848464995353, 0.2552893483045339, -0.32257868614889, 0.013839900264356116, 0.22680924478508585, 0.19606730442549097, -0.14893995214973374, -0.11862148625129211, 0.06280716897173126, 0.4404719011671693, 0.1848562335993722, -0.23766473320496317, 0.5080211360389013, 0.4896046093889448, 0.07882565971800044, 0.13807788510043884, 0.5501567523884221, 0.7459160398427234, 0.03929208763625259, 0.22750450492924412, 0.04832477386235923, 0.3939028245120492, 0.08390625716930607, -0.24088254962282987, 0.21055758730012417, -0.11940290750478755, -0.15184762711395403, -0.23689135719683646, 0.17859785496452765, -0.25486265135808545, 0.49159723035967284, -0.3085574375947232, 0.10449367553249156, -0.38403864213533706, 0.169757687485447, 0.08078529596215302, 0.11980233110084917, -0.3340576427045459, -0.6946829844476097, 0.24643698968039163, -0.452705501913013, -0.18040458269112686, 0.1614850360607852, -0.08760838149947534, 0.10152580706110363, -0.5134007570301459, 0.22876772361499645, 0.10281205190464378, 0.1738186912238501, 0.06538303693809339, 0.22154475774862356, -0.0564554495814543, 0.30562347514646615, -0.2920509162279544, -0.01325031550471584, 0.6895781582902385, 0.23064346979932926, 0.42426325686695654, -0.599425228500209, 0.4958148252464382, 0.018855405865355748, -0.3163622705312846, -0.036711028377043105, 0.48250881406677204, -0.022454084429045076, -0.13647134641624836, 0.332431250756777, 0.05913646915163013, 0.23915926064350612, 0.5563459109101732, 0.43027865602121024, -0.10575218174054184, -0.25024036769166424, -0.0828508493295143, 0.2885054877315155, 0.15897929143556264, 0.09570463590289738, -0.2510670937136365, -0.01985800640985858, 0.11096458082277164, -0.24300184381465945, 0.6244645281513294, -0.509788541753087, 0.37696439174576357, -0.21629545030721378, -0.11586552000722272, 0.032585972034558666, -0.0995403010409664, 0.18086949348774262, 0.2795187828691618, 0.2577772729854515, 0.5644417628771159, -0.38373385508101404, -0.14010722253269323, 0.1585117338367658, -0.18815583535662342, 0.04989884525785822, -0.2519346659294006, 0.0001550241739028374, 0.31466198099922915, -0.12186845097544391, 0.10001578613121899, -0.07071242863131555, -0.29703058521967546, -0.12806579701905005, -0.11244711093633432, -0.08089648721858184, 0.5816827704305054, 0.1686968101529935, 0.11535206231124633, -0.1934391261800256, -0.20757360485177603, 0.23965611866404424, -0.15198354639442033, -0.2397475093203286, -0.18627760656697914, -0.13094157692221445, 0.2913322675871335]
    #Will run in Webots for supervised observation and return fitness
    #fit = myController.evaluateRobot(individual)

main()
