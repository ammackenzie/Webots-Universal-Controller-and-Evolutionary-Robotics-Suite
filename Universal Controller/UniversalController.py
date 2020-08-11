from controller import Supervisor
from FixedNeuralNetwork import FixedNeuralNetwork
from RecurrentNeuralNetwork import RecurrentNeuralNetwork
from CMAES import CMAES, NCMAES
from MAPElites import MAPElites

import numpy as np
from DataReporting import DataReporting

'''
Universal supervisor controller for Webots Evolutionary Robotics experimentation
              
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
        self.neuralNetwork.decodeEA(individual) #individual passed from algorithm
        #note simulator start time
        t = self.supervisor.getTime()
        
        if mapElites:
            velocity = []
            angularV = []   
        
        #run simulation for evalRuntime seconds
        while self.supervisor.getTime() - t < self.evalRuntime:
            #calculate the motor speeds from the sensor readings
            self.computeMotors(self.getDSValues())
            #check current objective fitness
            currentFit = self.calculateFitness()
            
            if mapElites:
                currentV = self.robotNode.getVelocity()
                velocity.append(np.sqrt((currentV[0]**2) + (currentV[1]**2) + (currentV[2]**2)))
                angularV.append(np.sqrt(currentV[4]**2))
            
            #break if robot has reached the target
            if currentFit > self.solutionThreshold:
                timeTaken = self.supervisor.getTime() - t
                #safety measure due to timestep and thread lag
                if timeTaken > 0.0:  #then a legitimate solution
                    fit = self.calculateFitness()
                    break
            if self.supervisor.step(self.TIME_STEP) == -1:
                quit()
        
        #Get only the X and Y coordinates to create the endpoint vector
        endpoint = self.robotTrans.getSFVec3f()[0: 3: 2]
        distanceFS = np.sqrt((endpoint[0] - self.robotStartingLocation[0])**2 + (endpoint[1] - self.robotStartingLocation[2])**2)
        #reset the simulation
        self.resetAllPhysics()
        #find fitness
        fit = self.calculateFitness()
        if mapElites:
            averageVelocity = np.average(velocity)
            averageAngularV = np.average(angularV)
            return averageVelocity, distanceFS, fit
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
        
    def algorithmTest(self, myEA, generations, totalRuns, mapelites = False):
        self.dataReporting.algorithmTest(myEA, generations, totalRuns, mapelites)
    
    def controlGroupTest(self, generations = 10000, totalRuns = 1):
        self.dataReporting.controlGroupTest(individualSize=self.neuralNetwork.solutionSize, evalFunction = self.evaluateRobot, generations = generations, totalRuns = totalRuns)

def main():
    '''STEP 1: Create an array for your robot's distance sensors (names can be found in documentation or robot window)'''
    #distanceSensors = ['cs0', 'cs1', 'cs2', 'cs3']
    #Another example distance sensor array:
    distanceSensors = ['ds0', 'ds1', 'ds2', 'ds3', 'ds4', 'ds5']
    #distanceSensors = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
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
    #myEA = MAPElites(network.solutionSize, myController.evaluateRobot)
    myEA = MAPElites(network.solutionSize, evalFunction=myController.evaluateRobot, binCount = 10, tournamentSize = 5, maxVelocity = 0.125, maxDistance = 1.2)
    myEA.saveFile = "MAPElites easy race FNN 1KIP"
    #OPTIONAL FOR MAP ELITES - LOAD SAVED MAP
    #myEA.loadMap(myEA.saveFile)
    
    '''STEP 6: You can now run a test with your chosen set up'''
    #myEA.runAlgorithm(generations = 40)
    
    ''' ****************************************************************** '''
    '''OPTIONAL Data reporting 1: Create an instance of the DataReporting class with your desired filename '''
    #myData = DataReporting("H-CMAES FULL TEST - EscapeRoom, 1.0N, RNN, ")
    #myData = DataReporting("H -NCMAES EASY RACE")
    '''OPTIONAL Data reporting 2: Pass your DataReporting instance to your controller '''
    
    #myEA.runAlgorithm(generations = 200)
    '''OPTIONAL Data reporting 3: You can now run a recorded algorithm test '''
    #myController.algorithmTest(myEA, generations = 200, totalRuns = 10)
    
    for i in range(10):
        myData = DataReporting("Control Group Multi Maze Hemisson FNN SOURCE3" + str(i))
        myController.dataReporting = myData
        #myController.algorithmTest(myEA, generations = 10000, totalRuns = 1, mapelites=True)
        myController.controlGroupTest(generations = 10000, totalRuns = 1)
        #myEA.saveFile = "E-MAP EasyRace  1k IP FULL TESTING RNN"
        #myEA.runAlgorithm(generations = 10000)

   
    
    
    
    #myEA.saveFile = "E-MAP EasyRace  1k IP FULL TESTING RNN"
    #myEA.runAlgorithm(generations = 10000)

    ''' ****************************************************************** '''
    '''OPTIONAL Sigma DataVisualisationandTesting (compatible with CMA-ES and NCMA-ES: You can also run Sigma testing with your chosen set up'''
    #myData.sigmaGenerations = 6
    #myData.sigmaRuns = 25
    #myController.sigmaTest(myEA, upperLimit=2.0, lowerLimit= 0.1)
    
    ''' ****************************************************************** '''
    '''OPTIONAL MANUAL INDIVIDUAL TESTING: You can also manually test an array of individual NN weights outside of any algorithm'''
    #individual solution for easy race with e-puck 1mx1m Solution found! On eval: 5607
    #individual = [0.12945403736824845, 0.2661046512076877, -0.442799076276126, -0.25163068939132094, -0.39998322477695275, 0.4367100804250819, 0.14932849877684373, -0.10247225739024322, -0.28049880696625906, 0.05279904384511276, -0.20004441444544774, 0.12637396949644306, -0.09711332224080471, -0.06790293993134473, 0.6062531964690236, 1.0, 0.01199857899217141, 0.031088400765710483, -0.48669545840981965, -0.06477056580140331, 0.17348344410012487, -0.10205930908890434, 0.08103073737447701, 0.09576936120094612, -0.4464565851753052, 0.0259358780810231, -0.019213263304493466, 0.5640726955396176, -0.25295438727782715, 0.23042903435681655, 0.05066352575523572, -0.04432006353041552, 0.396756611080559, -0.3499859742015204, -0.45978909869518586, 0.47870808913298285, -0.04820786436406044, -0.10888952401405624, 0.23910209172609412, 0.11296819341897714, -0.15378412927873095, 0.02866985442078905, 0.601706445981838, 0.2132641291925935, 0.5841590612183303, 0.10287873267003281, -0.05575760659950829, -0.03039489650504712, -0.30056564658002816, 0.3686148695947607, 0.06813504048967779, 0.14657563898842724, 0.011891880158149223, 0.06352697168216735, -0.2250824593628234, 0.18480326174317765, -0.3737779485041332, 0.19832533753553583, 0.41456489966320026, 0.11336913656415089, 0.09800169279468787, -0.6371526551950911, 0.39426202221024853, -0.6397957190954771, 0.3788887929856996, -0.4936307579691912, 0.18055686937522877, -0.11433405241487263, -0.05909526530498005, -0.13246484397873068, 0.46147106020819467, -0.12690207237417186, 0.5025904882339401, -0.2242339084205368, -0.009331664112470273, 0.019899854883097687, -0.20449681272846357, -0.44435427399524996, -0.33798527895419755, -0.5792212797889922, 0.16341426866642886, 0.2199566214354949, -0.11680360783398126, -0.2858148484435559, -0.24268137588947658, 0.23897480808665378, 0.6734510396371823, 0.10902108238894953, -0.4926019477496155, 0.062065761329503394, 0.2636037226250653, 0.03763443585307785, -0.3359799093410882, -0.4299376661023095, -0.0005757769002054242, 0.1744444324196722, -0.09944970202962183, -0.015383170564508802, -0.1731596277882181, -0.835926961815893, -0.27098246698756623, -0.0212112901401921, -0.09466925889628855, -0.41282385456631543, -0.02806781038717559, 0.2975428505279796, -0.09374676122974314, 0.11169159799052467, 0.061867100328850844, 0.7270183949469506, 0.13246579022355556, -0.560921320078621, -0.5828797049100982, 0.4384047706290063, -0.05073308706423074, -0.025071563485543596, 0.22999949322300012, -0.3292594305019527, -0.12743425785594173, -0.2495183663166736, 0.1868502206258293, -0.29311117197126924, -0.6747315322796922, -0.005850927163965248, 0.555680740811277, 0.20723123960255094, -0.0922301906889343, -0.03326207663712094, 0.4061645550571117, -0.27581817195614444, -0.2552925288381096, 0.9622630678863899, -0.32712633607105607, 0.2241408686241418, 0.15203032617656068, -0.27630955571507043, -0.8195522131744277, -0.2525668947131633, 0.20279793935159365, 0.3191490250205179, 0.33108834761545064, -0.00601154947440137, -0.3239398636523605, 0.38216827813946297, 0.11553947059158974, 0.4169898903797664, 0.767311284377825, -0.2905349365445161, 0.17877940403879522, 0.13816671808980657, -0.12116449674896551, 0.2624316445856032, 0.2825229647668338, -0.05476575801003504, -0.07189881436976828, -0.2903849529745689, -0.21354383606670727, -0.1834542811673639, 0.3190759078930014, -0.32702519555235965, -0.029107329321433673, -0.22408925191846238, -0.13133422784337928, -0.4003158037212808, 0.36102695968087956, 0.22554502371944737, 0.3530085409564165, 0.9231016068017175, 0.07667557830815619, 0.01229332253493852, -0.07597197068402779, -0.06469036346859414, -0.14752930752412818, -0.05982452801335593, -0.38176333603906004, -0.47233262229285505, 0.03263853642723419, 0.029227855684033688, 0.08304088724648691, -0.03046623530322539, -0.017804319088514604, 0.42903757685490224, -0.031613076172143255, 0.2314488848554324, 0.5978070344484891, -0.038920962388328505, -0.21359637108639185, 0.45045268567418, -0.0740053839604044, -0.029199912627374786, 0.20727573361225646, 0.6338911021670377, 0.2582488226893089, -0.4040941398310275, -0.2530673226840402, 0.053066007620309275, 0.5903617685719513, 0.1886990726947373, -0.006042861072322575, 0.4374083912020556, 0.41994909663285934, -0.7748669716260062, -0.06273141849492735, 0.1898874728415736, -0.1233037117609255, 0.6405250879188488, -0.1880457474968089, -0.0138608766169553, 0.050950545867700564, 0.1115875490832049, 0.16716844206005932, 0.5344889614636321, -0.20844399724657053, 0.337052698873548, 0.3810063421657987, -0.2790316774530053, 0.5243103402226734, 0.3146338064393976, 0.06307183480426357, 0.20071538915705964, -0.19779913914735503, -0.4037205130609307, 0.0658195798164064, -0.15824833648940134, -0.03448551930740715, 0.11988950615375132, 0.2957861575280021, -0.1542955139210807, 0.5849660512184536, 0.1179430795156445, -0.16852828366029918, 0.3051751441305204, 0.5575193248080975, 0.329149138107467, 0.03509808839088466, 0.1011090564848662, 0.6616387658970025, -0.09631366060443963, 0.3848127550660378, 0.50650130999832, 0.23808109388387366, -0.013311898888700519, 0.21352424024344613, 0.055208352315055303, 0.2807574155380762, 0.1411115060195857, 0.6187854936619418, 0.07260600524874973, 0.2029032404816177, -0.19603659205048354, -0.06323452975287774, 0.22590948598972704, 0.3204527636609793, 0.4179100205108655]
    #fit = myController.evaluateRobot(individual)

main()
