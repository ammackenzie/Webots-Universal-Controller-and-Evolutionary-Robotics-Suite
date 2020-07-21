# Webots-Universal-Controller-and-Reinforcement-Learning-Suite

A universal supervisor controller and Reinforcement Learning suite for Webots ([https://www.cyberbotics.com/](https://www.cyberbotics.com/) ) that can be adapted to **any wheeled robot morphology** with ease. This is also a Reinforcement Learning suite that allows for easy experimentation and controller synthesis with the popular **CMA-ES** evolutionary algorithm, a **novelty search** augmented CMA-ES, and **MAP-Elites**, as well as both a fixed and recurrent neural network.

This suite allows the user to pick a desired configuration of Neural Network and Algorithm to carry out Reinforcement Learning with a robot of their choice. 

This is a **work in progress** and will continued to be refined, streamlined and expanded.

## **Coding Standards**

All code code has been designed for **ease of use** and **educational** purposes. Variables are as **close to plain English** as possible, and in cases speed and space efficiency has been sacrificed to make each step of the coding involved **as clear as possible** to the user.

This suite is designed to be used by non-technical as well as technical users and it is asked that the above coding standards are kept in mind during any future merge/pull requests. 

## **Universal Controller**
(Set up instructions can also be found as comments in each relevant module.)
*Features*
 - Default configuration is for a two wheeled robotic agent but can be easily expanded - instructions inside module next to relevant methods
 - Within a few lines of code a non-technical user can carry out Reinforcement Learning/Controller synthesis with a robotic agent and environment of their choice
 - Controller automatically:
	 - Enables and registers all desired distance sensors and motors
	 - Registers robot and target starting position, and resets all robot physics and translation in-between runs
	 - Tracks and evaluates robot's position and fitness during every time step and will break when a solution has been found
	 - Reads in distance sensor values and passes through Neural Network every time step
	 - Computes/updates all motor values via Neural Network output every time step

Example set up code from UniversalController.py:

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
    myEA.runAlgorithm(generations = 40)


## **Available Configurations:**

 *Neural Networks*
 - Fixed Neural Network - standard feed forward NN
 - Recurrent Neural Network - Elman Neural Network

*Algorithms*

 - CMA-ES - Deap Library Implementation
 - NCMA-ES - Novelty Search Augmented version of CMA-ES
 - MAP-Elites - Popular Quality Diversity Algorithm

*Morphologies*

 - Currently usable with any wheeled robot - default configuration is two wheels and can be manually edited via instructions in UniversalController.py
 

## **Data Reporting**

 - Dedicated DataReporting module with instructions that allows for:
	 - Saving and loading of all run and hyperparameter testing data
	 - Visualisation of run data via box plots
	 - Student's T Testing and P-value testing for Sigma(HyperParameter) testing data

Example DataReporting set up code from UniversalController.py:

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

Example data visualisation code from DataVisualisationandTesting.py:

    '''BOXPLOT Exmaple from Algorithm Test Data'''
    myData = DataReporting("Algorithm Testing Data/CMA-ES/CMA-ES TEST - MiddleWall - 150ME, 25PS, 20TR, 07-21-2020, 00-28-46.txt")
    myData.totalRuns = 20
    myData.maxEvals = 150
    myData.loadData()
    myData.displayAlgorithmBoxPlots("CMA-ES", "NCMA-ES")


A formal user guide will be put together in the future.


# **Known Deap Library Issue**

Deap library throws an exception that does not interrupt the running program  but displays in the console window when the CMA-ES algorithm resets itself in-between completed runs. You can simply comment out the few lines of code it refers to if you wish to avoid it clogging up the window.
