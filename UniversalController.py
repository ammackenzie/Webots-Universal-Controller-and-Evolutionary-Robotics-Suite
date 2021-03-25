from controller import Supervisor
from FixedNeuralNetwork import FixedNeuralNetwork
from RecurrentNeuralNetwork import RecurrentNeuralNetwork
from CMAES import CMAES, NCMAES, NIPES
from MAPElites import MAPElites

import numpy as np
from DataReporting import DataReporting

'''
Universal supervisor controller for Webots Evolutionary Robotics experimentation
              
QUICK START
To successfully initialise:
 - First, via the robot window or the Webots documentation, find the names of your robots distance sensors and place them into a string array like so: distance_sensors = ['cs0', 'cs1', 'cs2', 'cs3']
 - Now initialise desired type of Neural Network (fixed or recurrent) by initialising a new instance of desired class - i.e network = FixedNeuralNetwork(len(distance_sensors))
 - Create a new controller instance with desired values including passing the NN (robot and target names must be defined manually beforehand in DEF fields) - my_controller = UniversalController(network = network)
 - Pass your distance sensor array to your controller intance via the set_distance_sensors() function
 - Create instance of desired evolutionary strategy type (standard CMAES or novelty NCMAES) and pass desired solution size (taken from NN), reference to evaluation function my_EA, and desired population size like so: CMAES(individual_size = network.solution_size, eval_function = my_controller.evaluate_robot, pop_size = 20)
 - Call the runAlgorithm() method on your strategy instance
 
 - OPTIONAL - set custom space dimensions of your maze directly or with set_dimensions function (default is a 1m x 1m search space)
 
 MAP ELITES - REMIND them that they need to know what maxV and maxAV they can reach with their chosen robot
 
 DATA REPORTING GENERAL
  - Make new instance of DataReporting class and pass in the desired save file name (otherwise will be saved as "undefined.txt")
  - Then pass your DataReporting instance to your controller instance to store as a variable like: my_controller.dataReporting = my_data 
  - FOR SIGMA DataVisualisationandTesting you can now call the sigma_test(self, my_EA, upper_limit = 1.0, lower_limit = 0.1) function from your controller instance with desired range
  - SIGMA DataVisualisationandTesting is only compatible/necessary with an instance of the CMAES or NCMAES class and will use the pop_size already specified in your EA instance
  - SIGMA DataVisualisationandTesting is set to run 10 times for each sigma increment for 5 total generations based on pre set class variables in the DataReporting class - this can be changed if desired
  - Results will automatically be saved in a text file via pickle and input file names will automatically be appended with current date time - e.e "input 06-29-2020, 19-29-58.txt"
 DATA REPORTING ALGORITHM RUN
  - Once set up as per first two steps above, you can now call the algorithm_test function like so: my_controller.algorithm_test(my_EA, generations = 5, total_runs = 50)
  - These results will also be saved in a text file via pickle
 
 DATA REPORTING LOADING AND DISPLAYING DATA
  - Data can now be loaded and displayed for either sigma or algorithm DataVisualisationandTesting in a box plot using whatever IDE you wish
  - Simply create a new DataRerpoting class instance, passing the name of the file you wish to load like so: my_data = DataReporting("Sigma DataVisualisationandTesting - MiddleWall -  06-29-2020, 19-29-58.txt")
  - IF LOADING SIGMA DATA - first specify the lower and upper limits used like so: my_data.upper_limit = 1.0
                                                                                   my_data.lower_limit = 0.1
  - Now specify the maxEvals that were granted to each run (default is set to 100 in-class)
  - Now call the my_data.loadData() function
  - To display box plots now call either the my_data.displaySigmaBoxPlots() for SIGMA DataVisualisationandTesting results, or my_data.displayAlgorithmBoxPlots("CMA-ES") passing in the string/s you wish to be represented on the X axis
    EXAMPLE CODE FOR LOADING AND DISPLAYING SIGMA RESULTS
    my_data = DataReporting("Sigma DataVisualisationandTesting - MiddleWall - 100ME, 0.1 - 1.0, 06-29-2020, 19-29-58.txt")
    my_data.upper_limit = 1.0
    my_data.lower_limit = 0.1
    my_data.maxEvals = 100
    my_data.loadData()
    my_data.displaySigmaBoxPlots()
  
  - ADVANCED - multiple load files can be combined by simply calling the loadData function and passing in the name of the file you wish to load more data from. This can be used to compare multiple algorithms or experiments in one box plot automatically, like so:
    EXAMPLE CODE DISPLAYING BOX PLOTS FOR TWO RUNS OF CMAES
    my_data = DataReporting("Sigma DataVisualisationandTesting - MiddleWall -  06-29-2020, 19-29-58.txt")
    my_data.maxEvals = 100
    my_data.loadData()
    my_data.loadData("CMA-ES Test-MiddleWall-FNN07-16-2020, 16-52-07.txt")
    my_data.displayAlgorithmBoxPlots("CMA-ES", "CMA-ES")
    my_data.displayBoxPlots()
  
'''
class UniversalController:
    def __init__(self, network, robot_name = "standard", target_name = "TARGET", num_of_inputs = 4, eval_run_time = 100, time_step = 32):
        self.neural_network = network
        self.TIME_STEP = time_step #standard time step is 32 or 64
        self.supervisor = Supervisor()
        self.inputs = num_of_inputs
        self.solution_threshold = 0.95
        #call methods functions to intialise robot and target
        self.get_and_set_robot(robot_name)
        self.get_and_set_target(target_name)
        self.eval_run_time = eval_run_time
        
        #default to 1m x 1m space but can be edited directly or using method below
        self.max_distance = 1.4142
        
        self.data_reporting = DataReporting()
       
    def get_and_set_robot(self, robot_name):
        # initialise robot node and components
        self.robot_node = self.supervisor.getFromDef(robot_name)#TARGET ROBOT MUST BE NAMED IN DEF FIELD
        self.robot_trans = self.robot_node.getField("translation")
        self.robot_rotation = self.robot_node.getField("rotation")
        #call start location function to store initial position
        self.get_and_set_robotStart()
        #call motors function
        self.get_and_set_motors()
        
    def get_and_set_robotStart(self):
        #establish robot starting position
        self.robot_starting_location = self.robot_trans.getSFVec3f()
        self.robot_starting_rotation = self.robot_rotation.getSFRotation()
    
    '''EDIT HERE TO EXPAND TO DIFFERENT NUMBERS OF WHEELS/MOTORS'''
    def get_and_set_motors(self):
        self.motors = []
        self.motor_max_Vs = []
        #get robot motors - currently works for all two wheel motor morphologies
        self.left_motor = self.supervisor.getMotor('left wheel motor')
        self.right_motor = self.supervisor.getMotor('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        '''surveyor code'''
       # self.motors.append(self.supervisor.getMotor('wheel_motor00'))
        #self.motors.append(self.supervisor.getMotor('wheel_motor01'))
        #self.motors.append(self.supervisor.getMotor('wheel_motor02'))
       # self.motors.append(self.supervisor.getMotor('wheel_motor03'))
       # self.motors.append(self.supervisor.getMotor('wheel_motor04'))
       # self.motors.append(self.supervisor.getMotor('wheel_motor05'))
       # self.motors.append(self.supervisor.getMotor('wheel_motor06'))
        #self.motors.append(self.supervisor.getMotor('wheel_motor07'))
        #self.motors.append(self.supervisor.getMotor('wheel_motor08'))
       # self.motors.append(self.supervisor.getMotor('wheel_motor09'))
        
#         self.left_motor2 = self.supervisor.getMotor('wheel_motor01')
#         self.left_motor3 = self.supervisor.getMotor('wheel_motor02')
#         self.left_motor4 = self.supervisor.getMotor('wheel_motor03')
#         self.left_motor5 = self.supervisor.getMotor('wheel_motor04')
#         self.right_motor1 = self.supervisor.getMotor('wheel_motor05')
#         self.right_motor2 = self.supervisor.getMotor('wheel_motor06')
#         self.right_motor3 = self.supervisor.getMotor('wheel_motor07')
#         self.right_motor4 = self.supervisor.getMotor('wheel_motor08')
#         self.right_motor5 = self.supervisor.getMotor('wheel_motor09')
        #self.left_motor.setPosition(float('inf'))
       # self.right_motor.setPosition(float('inf'))
        
       # for i in range(len(self.motors)):
        #    self.motors[i].setPosition(float('inf'))
         #   self.motor_max_Vs.append(self.motors[i].getMaxVelocity())
        
        #get the max velocity each motor is capable of and set the max velocity
        self.left_motor_max = self.left_motor.getMaxVelocity()
        self.right_motor_max = self.right_motor.getMaxVelocity()
        
        #append necessary additional motors and max velocities here after enabling as above
        self.motors.append(self.left_motor)
        self.motors.append(self.right_motor)
        self.motor_max_Vs.append(self.left_motor_max)
        self.motor_max_Vs.append(self.right_motor_max)
    
    def get_and_set_target(self, target_name):
        #get target location
        self.target = self.supervisor.getFromDef(target_name)#TARGET MUST BE NAMED IN DEF FIELD
        #get and set the translation and location for retrieval when needed
        self.target_trans = self.target.getField("translation")
        self.target_location = self.target_trans.getSFVec3f()
        
    def set_dimensions(self, space_dimensions):
        #basic pythagorean calculation to find max distance possible in square space
        self.max_distance = np.sqrt(space_dimensions[0]**2 + space_dimensions[1]**2)
        #print(self.max_distance)
    
    def set_distance_sensors(self, distance_sensors):
        #takes in array of strings - names of distance sensors
        self.distance_sensors = []
        self.min_DS_values = []
        self.DS_value_range = []
        for i in range(len(distance_sensors)):
            #set and enable each sensor
            self.distance_sensors.append(self.supervisor.getDistanceSensor(distance_sensors[i]))
            self.distance_sensors[i].enable(self.TIME_STEP)
            #get and store the min reading value of each sensor
            self.min_DS_values.append(self.distance_sensors[i].getMinValue())
            #get and store the possible value range of each sensor
            self.DS_value_range.append(self.distance_sensors[i].getMaxValue() - self.min_DS_values[i])
        
        #print(self.DS_value_range)
    
    
    def get_DS_values(self):
        #get distance sensor values
        values = []
        for i in range(len(self.distance_sensors)):
            value = self.distance_sensors[i].getValue()
            #value = value/self.maxDSReading #practical max value
            value = value - self.min_DS_values[i]
            if value < 0.0:
                value = 0.0 #to account for gaussian noise
            value = value/(self.DS_value_range[i])
            #account for gaussian noise providing higher than max reading
            if value > 1.0:
                value = 1.0
            values.append(value)
        #return a list of the normalised sensor readings
        return values

    def compute_motors(self, DS_values):
        #get the outputs of the neural network and convert into wheel motor speeds
        #already fully flexible for multiple motors
        results = self.neural_network.forwardPass(DS_values) 
        for i in range(len(self.motors)):
            self.motors[i].setVelocity(results[i]*self.motor_max_Vs[i])
            #if(i < 2):
             #   self.motors[i].setVelocity(results[0]*self.motor_max_Vs[i])
            #else:
            #    self.motors[i].setVelocity(results[1]*self.motor_max_Vs[i])
            
            

    
    def reset_all_physics(self):
        #reset robot physics and return to starting translation ready for next run
        self.robot_rotation.setSFRotation(self.robot_starting_rotation)
        self.robot_trans.setSFVec3f(self.robot_starting_location)
        self.robot_node.resetPhysics()
    
    '''SIMULATION FUNCTION - can also be used directly for manual DataVisualisationandTesting of weight arrays (to manually repeat successful solutions etc.)'''
    def evaluate_robot(self, individual, map_elites = False, all_novelty = False):
        self.neural_network.decodeEA(individual) #individual passed from algorithm
        #note simulator start time
        t = self.supervisor.getTime()
        
        if map_elites or all_novelty:
            velocity = []
            angular_V = []   
        
        #run simulation for eval_run_time seconds
        while self.supervisor.getTime() - t < self.eval_run_time:
            #calculate the motor speeds from the sensor readings
            self.compute_motors(self.get_DS_values())
            #check current objective fitness
            current_fit = self.calculate_fitness()
                        
            if map_elites or all_novelty:
                currentV = self.robot_node.getVelocity()
                velocity.append(np.sqrt((currentV[0]**2) + (currentV[1]**2) + (currentV[2]**2)))
                angular_V.append(np.sqrt(currentV[4]**2))
            
            #break if robot has reached the target
            if current_fit > self.solution_threshold:
                time_taken = self.supervisor.getTime() - t
                #safety measure due to time_step and thread lag
                if time_taken > 0.0:  #then a legitimate solution
                    fit = self.calculate_fitness()
                    break
            if self.supervisor.step(self.TIME_STEP) == -1:
                quit()
        
        #Get only the X and Y coordinates to create the endpoint vector
        endpoint = self.robot_trans.getSFVec3f()[0: 3: 2]
        distance_FS = np.sqrt((endpoint[0] - self.robot_starting_location[0])**2 + (endpoint[1] - self.robot_starting_location[2])**2)
        #reset the simulation
        self.reset_all_physics()
        #find fitness
        fit = self.calculate_fitness()
        if map_elites:
            average_velocity = np.average(velocity)
            #average_angular_V = np.average(angular_V)
            return average_velocity, endpoint, distance_FS, fit
        
        if all_novelty:
            average_velocity = np.average(velocity)
            median_velocity = np.median(velocity)
            #print("fit = " + str(fit) + ", endpoint = " + str(endpoint[0]) + "," + str(endpoint[1]) + ", AV = " + str(average_velocity) + ", distanceFS = " + str(distance_FS))
            
            return fit, endpoint, average_velocity, distance_FS, median_velocity
        return fit, endpoint
        
    def calculate_fitness(self):
        values = self.robot_trans.getSFVec3f()
        distance_from_target = np.sqrt((self.target_location[0] - values[0])**2 + (self.target_location[2] - values[2])**2)
        fit = 1.0 - (distance_from_target/ self.max_distance)
        return fit
    
    def sigma_test(self, my_EA, upper_limit = 1.0, lower_limit = 0.1):
        self.data_reporting.sigma_test(my_EA, upper_limit, lower_limit)
        
    def algorithm_test(self, my_EA, generations, total_runs, nipes=False, map_elites = False):
        self.data_reporting.algorithm_test(my_EA, generations, total_runs, nipes, map_elites)
    
    def control_group_test(self, generations = 10000, total_runs = 1):
        self.data_reporting.control_group_test(individual_size=self.neural_network.solution_size, eval_function = self.evaluate_robot, generations = generations, total_runs = total_runs)

def main():
    '''STEP 1: Create an array for your robot's distance sensors (names can be found in documentation or robot window)'''
    distance_sensors = ['cs0', 'cs1', 'cs2', 'cs3']
    #Surveyor DS
    #distance_sensors = ['ds0', 'ds1', 'ds2', 'ds3']
    #Another example distance sensor array:
    #distance_sensors = ['ds0', 'ds1', 'ds2', 'ds3', 'ds4', 'ds5']
    #distance_sensors = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
    '''STEP 2: Create your Neural Network instance '''
    #network = FixedNeuralNetwork(inputs = len(distance_sensors))
    network = RecurrentNeuralNetwork(inputs = len(distance_sensors))
    '''STEP 3: Create your controller instance (pass in the network) '''
    my_controller = UniversalController(network = network)
    #optional - default is set to 100 seconds
    my_controller.eval_run_time = 100

    '''STEP 4: Pass your distance sensor array to your controller'''
    my_controller.set_distance_sensors(distance_sensors)
    #optional - set size of your environment, default is 1m x 1m
    #my_controller.set_dimensions([0.75, 0.75])
    
    '''STEP 5: Create your algorithm instance '''
    #my_EA = CMAES(individual_size = network.solution_size, eval_function = my_controller.evaluate_robot, pop_size = 50, sigma=1.0)
    #my_EA = NCMAES(individual_size = network.solution_size, eval_function = my_controller.evaluate_robot, pop_size = 50)
    #my_EA.switch_to_average_velocity()
    #my_EA.switch_to_distance_FS()
    #my_EA = MAPElites(network.solution_size, my_controller.evaluate_robot)
    #my_EA = MAPElites(network.solution_size, eval_function=my_controller.evaluate_robot, bin_count = 10, tournament_size = 5, max_velocity = 0.125, max_distance = 1.2)
    #my_EA.saveFile = "TEST"
    #OPTIONAL FOR MAP ELITES - LOAD SAVED MAP
    #my_EA.loadMap(my_EA.saveFile)
    
    '''STEP 6: You can now run a test with your chosen set up'''
    #my_EA.runAlgorithm(generations = 40)
    
    ''' ****************************************************************** '''
    '''OPTIONAL Data reporting 1: Create an instance of the DataReporting class with your desired filename '''
    #my_data = DataReporting("NIPES TEST - EasyRace, RNN ")
    #my_data = DataReporting("H -NCMAES EASY RACE")
    '''OPTIONAL Data reporting 2: Pass your DataReporting instance to your controller '''
    
    #my_EA.runAlgorithm(generations = 200)
    '''OPTIONAL Data reporting 3: You can now run a recorded algorithm test '''
    #my_controller.algorithm_test(my_EA, generations = 200, total_runs = 10)
    
    #max distance for two rooms = 1.0185
    #my_controller.max_distance=1.0185
    
    #HARD MAZE MAX DISTANCE = 1.247
    #my_controller.max_distance=1.247
    
    #HARD RACE MAX D
    my_controller.max_distance=1.2
    
 #==============================================================================
    for i in range(10):
        #my_EA = NCMAES(individual_size = network.solution_size, eval_function = my_controller.evaluate_robot, pop_size = 50)
        #my_EA.switch_to_average_velocity()
        #my_EA.switch_to_distance_FS()
        #my_EA = MAPElites(network.solution_size, eval_function=my_controller.evaluate_robot, bin_count = 10, tournament_size = 5, max_velocity = 0.125, x_values=[-0.465, 0.465], y_values=[-0.465, 0.465])
        my_EA = NIPES(individual_size = network.solution_size, eval_function = my_controller.evaluate_robot, pop_size = 10)
        my_EA.solution_threshold  = .90
        my_controller.solution_threshold  = .90
        #my_EA.switch_to_average_velocity()
        #my_EA.switch_to_distance_FS()
        #my_EA.switch_to_median_velocity()
        my_EA.switch_to_multi_dimensional_descriptor([1, 2])
        my_data = DataReporting("(HR)rerun TEST MAP-ELITES AV-EP  .90ST " + str(i))
        my_controller.data_reporting = my_data
         
        #10k max evals for nipes - does not take in gens
        my_controller.algorithm_test(my_EA, generations = 10000, total_runs = 1, nipes=True, map_elites=False)
        #my_controller.algorithm_test(my_EA, generations = 20, total_runs = 1, nipes=False, map_elites=True)
        #my_controller.control_group_test(generations = 10, total_runs = 1)
        #my_EA.saveFile = "E-MAP HardRace  1K IP FULL TESTING RNN"
        #my_EA.run_algorithm(generations = 200)
  
    my_data.save_overall_test_results()
 #==============================================================================
    #NIPES SOLUTION TO HARDRACE
    #[-1.0, -1.0, -1.0, -0.04267673089265058, 1.0, 0.8711768242989855, 0.37712015679160116, -0.40659511243056934, -0.07759068099424195, -0.06483330592542512, 0.6446643123893196, 0.8191767620302154, -0.14023726491367455, 0.6101015843588247, 0.8248399492575658, 0.633405589509698, -0.9815713601906813, 0.0406070324414159, -1.0, 1.0, 0.8073935202335807, 1.0, -0.5350219489508312, 0.4950030018121688, 1.0, -1.0, -1.0, 0.23385220010609312, -0.9328204807399313, 0.004179965962517906, 1.0, -0.604280342424324, -1.0, 0.8662357894651285, -1.0, 1.0, 1.0, -1.0, 0.9175341461978542, -0.7371416888672844, -1.0, 1.0, -1.0, 0.13144437786946753, 0.43814450942024163, -1.0, 0.25651687490723646, -0.7562405465595007, 0.5859198559426336, -1.0, 0.6500824858480528, 0.6929510911449608, 1.0, -1.0, 0.8626071115189453, 1.0, 1.0, -1.0, -1.0, 0.832327595419298, 0.6770498558967191, -0.2542732901888225, 0.005692038512526908, -0.3954563446283999, -0.26487935174952376, 1.0, -0.1599855501263171, 1.0, 0.36656584982035656, 0.5774061281375241, -0.4877536746820789, -1.0, -1.0, 0.18323533253072208, 1.0, 0.2933242938821475, 0.8656966692786565, -0.1973061085791515, -0.721506147735537, -0.29410075444894307, 1.0, 1.0, 1.0, -1.0, 1.0, -0.7943490540267687, 0.36398280295924446, 1.0, 0.05788776545087352, 1.0, 0.1674565821804713, -1.0, 0.060320351713348805, 1.0, -0.3196553533244266, -0.6846910300968655, -0.3513805237835505, -0.4942863547291071, 0.1351163528506432, -0.1277346984989595, -0.9204147505600934, -1.0, -0.4475432144621061, 0.82219790174153, 1.0, 0.2017541766181132, 0.5119884509299886, 0.7000883764077617, -0.6776554846382423, -1.0, -0.9495090461769615, -0.24937195816699054, 0.5983529529008137, -0.1945363463465341, 1.0, -1.0, -1.0, 0.989558672006314, 0.09731698299514016, -1.0, -0.41579985464499075, 0.3127037763017473]
    
    #90% nipes solutions to try later
    #[-0.22814387576141365, 0.6865956844996054, -0.07602685804702639, 0.9386583655546602, 1.0, 1.0, -0.8081637510010959, 1.0, -0.13839597353163585, 0.1386767898495507, 1.0, -0.37197630459278974, -0.5967225137360288, -0.7223291283848469, 0.21026403083338688, -0.6122092211933444, 0.6329978632716164, 0.6420229902650141, -0.6854702614721718, -1.0, -1.0, 0.06533873128393536, -1.0, 0.133286605384179, -0.6939657521368885, 0.998061028086333, 0.9611764609788924, -0.9082115540240299, -0.2738947292472498, -0.5890163622936122, -1.0, -0.2273146699488455, -0.34857816443130424, 0.5457332111404277, 1.0, -0.5934264679420684, -0.8381836218527375, -0.7740614057513644, 0.7541247789465338, 1.0, -0.9029838838290749, -0.19300741098249818, -0.9951683566694028, 0.9628250587500877, 1.0, -1.0, 0.742306712840802, 0.23448426298550246, 1.0, -0.018637968568317, -0.07021912658331068, 1.0, -1.0, 0.4295170188095053, -1.0, -0.5340737760081214, 1.0, 0.6106739783504682, -1.0, 0.4386934469745415, -1.0, -0.22476657634306332, 0.2957295489620128, 1.0, 0.8863984135764961, 0.4629381762727698, 1.0, -0.0812423696750036, -0.06871366639369275, -1.0, -1.0, -0.5646153851610816, 0.7709890467551151, 0.39930089607438995, -0.22288317113426648, -0.2604570048058523, -1.0, 1.0, 1.0, -0.6098724098040218, -1.0, 1.0, -0.45325662003749806, 0.18062713534415334, 1.0, -0.006191628958297912, -0.21764269705339373, -0.733678393274307, 1.0, -1.0, -0.5195190776589349, -0.2636862421911325, 1.0, -1.0, -0.336470542358047, -0.25049564226178217, 1.0, -0.02325124072746662, -1.0, -0.4890043269256427, -0.4139717172174276, 1.0, 0.5599423347412702, -0.10812186286596054, -1.0, 0.7472192930992847, -0.10399075503885567, 0.901421719848227, 1.0, -0.3710137918170247, -1.0, -0.5573819606443515, -0.9726737498365602, -0.23502204379332192, -1.0, -0.674990455064638, 0.27091598329057254, -0.7914948435017227, 1.0, 1.0, -0.708613960436458, -1.0]
    
    #[-1.0, -0.5202682038981028, -0.07223557824662846, -0.11541568399200475, 0.9469281776756433, -0.35062550691331856, 1.0, 1.0, -1.0, 1.0, -1.0, 0.4401340801703167, -0.6658036605463955, 0.22737660864874124, -0.8476295718706739, -0.7340489344801506, -0.7963308952870819, -1.0, 0.45450442448516565, -0.4128799659813492, 0.5181234016673935, -1.0, 0.5062792876365728, 0.7799413314465048, 0.7849396276249171, 1.0, -1.0, 0.500748599665213, -0.4813044293678155, -0.09469311412607546, 1.0, 0.8126248514965939, -0.11277794737544221, 1.0, 0.5752347456398694, -1.0, -1.0, 0.880624968925204, -0.2669124525380955, -1.0, 1.0, -0.057352756614834494, 1.0, -0.0813099617191182, -0.027215859876297804, 1.0, 0.5415565822687354, 0.6656472873971415, -1.0, -0.29489813909324114, -0.5903735390457902, 0.6479932338230684, 1.0, -0.507651126768197, -0.8214402512620598, -1.0, -0.6431004085733087, 0.9249346310162206, 0.45471362015343253, 0.16323152727339985, 0.3387108879399108, 0.21490761190031707, 1.0, -0.8061127752526424, -0.637289379709163, 1.0, -0.5059434939236636, 0.775745432260962, 0.8991347167113303, 0.4029676016072172, -0.6912568429210759, -0.9997088824262612, -0.22986442433967144, 0.9201019783437591, 0.5896035795180227, -0.531336255661328, -0.7814624188274459, 0.8474426856027149, 0.20606176284292982, -1.0, 0.5412034072280367, 0.3572424859764568, -0.82884882072761, 0.7496028438353938, 0.3558053953215643, -0.6251048014555677, 0.6153170036563149, 0.5395370738295486, 0.6385350180455273, 0.4746317761344241, 0.6016057136277925, 1.0, 0.8974312841126039, 0.6031630055124831, 0.19579047007401554, 0.9063494832169676, 0.42627972996820357, 0.33712123046664194, 0.04061933713999756, 1.0, -0.5850974689895054, 1.0, 1.0, 0.368547858286106, -0.3040771064960058, -0.8234936866901892, 0.5373656622964263, 0.5982646809591108, -0.6048471059404875, -0.031226602676914888, -0.38540933346439565, -0.42000116952485245, 0.24876265440684123, -1.0, 0.10255283246957995, -1.0, -0.17331081872663995, -1.0, 1.0, -1.0, 1.0, 0.061786244170900124]
    #[1.0, 0.13959259895618598, 0.6246523567350016, -0.8544701601072711, -1.0, -0.23003877164058847, -1.0, -1.0, 0.47043203722868276, 1.0, 0.5341429879010202, -0.7539790447235581, -0.8549908779351302, -1.0, -0.9490494827657102, 0.4117245952732415, 0.20479438858929588, -0.641321291024519, 0.2577429636444925, 0.5908781547788096, -0.6595646978685219, 0.4117117813328713, 0.10267318112680646, 0.4009204973003082, -0.08590140849674885, 1.0, 1.0, 0.12652925720909985, 0.18750466873751712, 1.0, 0.34259815074334893, -1.0, -0.24690949678472374, -0.40561125698429595, 1.0, -1.0, 1.0, -1.0, 0.5723847958486407, 0.7149219696243441, -0.34623634442129597, 0.6530563208055468, -0.627986304872186, 0.3243132873506441, -1.0, 0.40208115994581745, -0.2966387354612757, 0.019767158074273608, -1.0, 0.5770908489690131, -0.30417122031123794, -0.48640681616419745, -1.0, -0.04834559461161166, -0.5208283918190575, -0.5931563939712023, -0.5472740850131352, 0.970138011461012, -1.0, -0.7585600138061025, 0.3014411201403495, -1.0, 0.45196187135482124, -1.0, 0.6452854239000663, -1.0, 1.0, -1.0, -0.31737317002778587, -0.05837233317426745, -0.836451116911058, 1.0, -0.9422334148035697, -0.36107297420784545, -0.4593419685661435, 0.07225172542635146, 1.0, -1.0, 0.47741233422690715, 0.9434750033943367, 0.13995881627200318, -0.5520579891557081, 0.026265534920110314, 1.0, -0.6166418444779022, 0.3364133174707358, 0.8734777839389479, 0.45582218694924126, -1.0, 0.5072532480727587, 1.0, -1.0, 0.579399489690028, 0.10023148679396919, 0.21778765502178743, 1.0, 1.0, 1.0, 0.17767804244568894, -0.15677733884432932, 1.0, -0.5377839219542052, 1.0, -1.0, -0.8754554028803896, 0.6008434419628147, -0.9687564115345257, 1.0, 1.0, 0.48430979554848136, -0.875708499818095, -1.0, 0.4477821664739284, -0.35848444597920337, -1.0, 1.0, 0.8270180376874101, -1.0, -0.7851851986434667, -0.8708422675679494, -1.0, 0.750488898347355]
    # [0.07748108901508075, 0.0543144872200746, 0.611800944572926, 1.0, -0.4232651306933256, 0.4633705420590669, 0.24573813342989603, 0.42821429281603957, 1.0, 1.0, -1.0, 0.6099860660379569, -0.027302629362711363, -0.41261168829875766, -1.0, -0.9725859383685364, -0.23299452337927706, 0.35659011708975136, 0.1449830838933087, -0.8039890939559334, 0.14290597034050295, 1.0, 0.6975723905398891, -0.16813800621070524, 1.0, 0.7571321969466553, 1.0, -0.4326151848813862, -0.39580271542183043, -1.0, 0.3517221868661927, 0.5548223214560957, 1.0, -1.0, -1.0, -0.3937271440548745, -0.99396049772706, -0.7671313612673035, -0.639623656692509, 1.0, 0.38346242144921766, 0.43003545925221726, 0.33965551911455716, -1.0, 0.5243135046952802, 0.26286158237830803, -0.36917440570932064, -1.0, 0.84560076839853, -0.6456349237705201, 0.9902152415314391, -1.0, -0.975200362901769, -1.0, -0.6955401684024795, 0.38425157381727937, 0.8650017928799958, 0.40744066591598443, 0.7246044803141072, 1.0, -0.009131542219351809, 0.6256923735742596, -1.0, 0.44720500548886993, -1.0, 1.0, -0.8501684158353914, 0.9944306605391846, -0.7616331808299206, -1.0, 1.0, -0.6755295842957614, 0.9641301045997696, 0.28131393807855676, -0.6197858247872049, -1.0, -0.2256505128554313, 0.48803924710177654, 0.25224959156876875, -1.0, -0.11859409435378594, -1.0, 0.640998719995939, -1.0, 0.40920475455447175, 0.931535061483316, -0.854237749059888, -1.0, 1.0, -0.3215059793449326, 1.0, 0.18990859072107058, -0.4520285084180654, 1.0, 0.04544261825964323, -1.0, 0.2207687796518433, 0.2675645373813212, -1.0, 0.12955101406359212, -1.0, -0.09245816427602059, 1.0, 1.0, -0.7553557147919263, 1.0, -0.6147653608786289, -1.0, 0.1577323475026284, -0.08669451727109602, 1.0, 0.937545075117995, -0.9832700045392053, 0.8357625207734796, -0.6368602318875881, -1.0, 0.5032467393557291, -0.9844009884084205, -0.7596838836271143, 1.0, -1.0, -0.034405762633260645]
    #[0.9289540336316001, 0.21903552929840517, -1.0, -1.0, 0.6730624444463168, -0.5343068871373705, 0.004534679498982617, 0.3386671029625, -1.0, 1.0, -0.09639777596454889, -0.1705920374311611, -1.0, -1.0, -1.0, 0.4736717895282517, -1.0, 0.3088579111600518, -0.9045702868198413, -0.2897324975703127, 0.8033659612916512, 1.0, 0.523435590179999, 0.8490707003178031, -1.0, 1.0, 1.0, 0.4705016911958809, -0.3584745073993276, 1.0, -0.15752355475510657, -1.0, -0.12346028682596225, -0.03719296376032515, 0.09751764352384744, 1.0, -0.23787017309371508, -1.0, -0.07396833021698815, -0.89896999723323, 0.5389371098364627, -1.0, 0.5376666818050891, -1.0, -0.7406936074337837, -0.22399958711176635, -1.0, -0.6179050953273378, 0.40039510616902685, 0.3780004001555245, 1.0, -1.0, -0.12004315281548991, 1.0, -0.1890561381692688, -1.0, -0.8158610323540602, 1.0, 0.058461215633563636, -1.0, 0.24497560003240618, 0.7043862802702044, -0.47200871004318756, -0.16226771829660747, 1.0, -1.0, -1.0, 0.8845846955374059, 0.8849636719818894, 0.11495152662596189, 0.6422747269954064, -0.008766305272617134, -0.2325184673347717, 1.0, 0.8861465635184538, 0.3369607658021087, 1.0, -1.0, 0.21511584567803307, 0.4461353288240838, -1.0, 0.2387453587464628, 1.0, -0.722856188982131, 1.0, 0.6338503670220622, -0.0680203624376522, -0.23360007181351514, -0.42577524343414663, -1.0, -1.0, -0.5103755765055974, 0.08049415188843963, -0.8294103898439216, -1.0, -0.17751707299391736, 0.4573898253897542, -1.0, -0.17523026035156397, 0.9555520727223472, -0.9844130084132633, 0.37637803145279985, 1.0, -0.12743329546985563, -0.08704748827909832, -0.903436060212798, -0.2898361079852952, 1.0, -1.0, 1.0, -0.3268252040308846, 1.0, 0.0702091233483488, 0.8880825515198826, -1.0, 0.31950621799691764, -0.6032839923852331, 1.0, -0.4159000665798207, -0.26291006640874287, -0.7297113789291975, -0.7195276160629186]
    #[1.0, 1.0, 0.12939786472244694, 1.0, -1.0, -0.20964772477423269, -1.0, -0.03304427448129183, -0.3108537710447907, 0.4835876301967241, -0.3831502446335347, -0.2869994223243308, -1.0, 0.5654352266054201, -0.09842930362529312, 0.2285563714852365, 0.7569341456697329, -0.669899197838543, -0.8611131877791816, -1.0, -0.18667369726105953, 0.11334247806546346, 0.28115545527095004, 0.0059067429261868845, 1.0, 1.0, -1.0, 0.6655374986519105, -0.3626592669715874, 0.5227015982139332, -1.0, 1.0, 0.9085087866100618, -0.6929600695082768, -0.11936594957050108, 1.0, -1.0, 0.1692663757956894, 0.37159725742198424, -0.15470015572551418, 0.086770569381332, -0.7009079534186885, -1.0, -0.7455943478851343, 0.30583178181771314, -0.8362432814970214, -0.2978525081913632, -0.010206304100825216, 1.0, 0.7190629508367787, 0.4675047017310718, -1.0, 0.5255508068645302, -1.0, 1.0, -0.06696748954089682, 1.0, 1.0, 0.4485174937275693, -1.0, -0.7963722762298653, -1.0, -0.9445573529622361, 0.5744118558656698, 0.5798643703813261, -0.4650122409117762, 0.3716621121394059, 0.6703777279834047, 0.600162850070542, -1.0, 1.0, 1.0, 1.0, 0.2398572917376598, 0.37071353658743816, -1.0, -0.2085163881953626, -1.0, -0.21522725227504116, 1.0, -0.026611771693887693, -1.0, -0.25627680052084073, -0.4465089235119691, 1.0, -1.0, -1.0, 0.26703792121190517, -0.46689563458083316, 0.6155567117493931, 1.0, 0.20237462104624251, -0.1468619402038431, -0.09506651837411519, 0.1770441489556714, -0.6197659216791095, -1.0, 0.8441530546963046, 1.0, -0.5041795666483571, 0.4092219233822091, -0.7315087299722458, 0.30107866821129253, -0.4161374974258944, -0.9435265002365134, -0.925978558538969, 1.0, 0.5970503326440927, -1.0, -0.40916373029625625, 0.3872519763427434, -0.30128683096711384, 0.4873538436009186, 0.41895023055539965, 0.18158048323316725, 0.016944618849871793, 0.3859205654682292, 0.06292136782936682, -1.0, -1.0, -0.7385540488908866, 0.5103867193793262]
    
    
    #my_EA.saveFile = "E-MAP EasyRace  1k IP FULL TESTING RNN"
    #my_EA.runAlgorithm(generations = 10000)

    ''' ****************************************************************** '''
    '''OPTIONAL Sigma DataVisualisationandTesting (compatible with CMA-ES and NCMA-ES: You can also run Sigma testing with your chosen set up'''
    #my_data.sigmaGenerations = 6
    #my_data.sigmaRuns = 25
    #my_controller.sigma_test(my_EA, upper_limit=2.0, lower_limit= 0.1)
    
    ''' ****************************************************************** '''
    '''OPTIONAL MANUAL INDIVIDUAL TESTING: You can also manually test an array of individual NN weights outside of any algorithm'''
    #individual solution for easy race with e-puck 1mx1m Solution found! On eval: 5607
#     individual = [1.0, 1.0, 0.12939786472244694, 1.0, -1.0, -0.20964772477423269, -1.0, -0.03304427448129183, -0.3108537710447907, 0.4835876301967241, -0.3831502446335347, -0.2869994223243308, -1.0, 0.5654352266054201, -0.09842930362529312, 0.2285563714852365, 0.7569341456697329, -0.669899197838543, -0.8611131877791816, -1.0, -0.18667369726105953, 0.11334247806546346, 0.28115545527095004, 0.0059067429261868845, 1.0, 1.0, -1.0, 0.6655374986519105, -0.3626592669715874, 0.5227015982139332, -1.0, 1.0, 0.9085087866100618, -0.6929600695082768, -0.11936594957050108, 1.0, -1.0, 0.1692663757956894, 0.37159725742198424, -0.15470015572551418, 0.086770569381332, -0.7009079534186885, -1.0, -0.7455943478851343, 0.30583178181771314, -0.8362432814970214, -0.2978525081913632, -0.010206304100825216, 1.0, 0.7190629508367787, 0.4675047017310718, -1.0, 0.5255508068645302, -1.0, 1.0, -0.06696748954089682, 1.0, 1.0, 0.4485174937275693, -1.0, -0.7963722762298653, -1.0, -0.9445573529622361, 0.5744118558656698, 0.5798643703813261, -0.4650122409117762, 0.3716621121394059, 0.6703777279834047, 0.600162850070542, -1.0, 1.0, 1.0, 1.0, 0.2398572917376598, 0.37071353658743816, -1.0, -0.2085163881953626, -1.0, -0.21522725227504116, 1.0, -0.026611771693887693, -1.0, -0.25627680052084073, -0.4465089235119691, 1.0, -1.0, -1.0, 0.26703792121190517, -0.46689563458083316, 0.6155567117493931, 1.0, 0.20237462104624251, -0.1468619402038431, -0.09506651837411519, 0.1770441489556714, -0.6197659216791095, -1.0, 0.8441530546963046, 1.0, -0.5041795666483571, 0.4092219233822091, -0.7315087299722458, 0.30107866821129253, -0.4161374974258944, -0.9435265002365134, -0.925978558538969, 1.0, 0.5970503326440927, -1.0, -0.40916373029625625, 0.3872519763427434, -0.30128683096711384, 0.4873538436009186, 0.41895023055539965, 0.18158048323316725, 0.016944618849871793, 0.3859205654682292, 0.06292136782936682, -1.0, -1.0, -0.7385540488908866, 0.5103867193793262]
#     fit = my_controller.evaluate_robot(individual)
#     individual = [-0.22814387576141365, 0.6865956844996054, -0.07602685804702639, 0.9386583655546602, 1.0, 1.0, -0.8081637510010959, 1.0, -0.13839597353163585, 0.1386767898495507, 1.0, -0.37197630459278974, -0.5967225137360288, -0.7223291283848469, 0.21026403083338688, -0.6122092211933444, 0.6329978632716164, 0.6420229902650141, -0.6854702614721718, -1.0, -1.0, 0.06533873128393536, -1.0, 0.133286605384179, -0.6939657521368885, 0.998061028086333, 0.9611764609788924, -0.9082115540240299, -0.2738947292472498, -0.5890163622936122, -1.0, -0.2273146699488455, -0.34857816443130424, 0.5457332111404277, 1.0, -0.5934264679420684, -0.8381836218527375, -0.7740614057513644, 0.7541247789465338, 1.0, -0.9029838838290749, -0.19300741098249818, -0.9951683566694028, 0.9628250587500877, 1.0, -1.0, 0.742306712840802, 0.23448426298550246, 1.0, -0.018637968568317, -0.07021912658331068, 1.0, -1.0, 0.4295170188095053, -1.0, -0.5340737760081214, 1.0, 0.6106739783504682, -1.0, 0.4386934469745415, -1.0, -0.22476657634306332, 0.2957295489620128, 1.0, 0.8863984135764961, 0.4629381762727698, 1.0, -0.0812423696750036, -0.06871366639369275, -1.0, -1.0, -0.5646153851610816, 0.7709890467551151, 0.39930089607438995, -0.22288317113426648, -0.2604570048058523, -1.0, 1.0, 1.0, -0.6098724098040218, -1.0, 1.0, -0.45325662003749806, 0.18062713534415334, 1.0, -0.006191628958297912, -0.21764269705339373, -0.733678393274307, 1.0, -1.0, -0.5195190776589349, -0.2636862421911325, 1.0, -1.0, -0.336470542358047, -0.25049564226178217, 1.0, -0.02325124072746662, -1.0, -0.4890043269256427, -0.4139717172174276, 1.0, 0.5599423347412702, -0.10812186286596054, -1.0, 0.7472192930992847, -0.10399075503885567, 0.901421719848227, 1.0, -0.3710137918170247, -1.0, -0.5573819606443515, -0.9726737498365602, -0.23502204379332192, -1.0, -0.674990455064638, 0.27091598329057254, -0.7914948435017227, 1.0, 1.0, -0.708613960436458, -1.0]
#     fit = my_controller.evaluate_robot(individual)
#     individual = [1.0, 0.13959259895618598, 0.6246523567350016, -0.8544701601072711, -1.0, -0.23003877164058847, -1.0, -1.0, 0.47043203722868276, 1.0, 0.5341429879010202, -0.7539790447235581, -0.8549908779351302, -1.0, -0.9490494827657102, 0.4117245952732415, 0.20479438858929588, -0.641321291024519, 0.2577429636444925, 0.5908781547788096, -0.6595646978685219, 0.4117117813328713, 0.10267318112680646, 0.4009204973003082, -0.08590140849674885, 1.0, 1.0, 0.12652925720909985, 0.18750466873751712, 1.0, 0.34259815074334893, -1.0, -0.24690949678472374, -0.40561125698429595, 1.0, -1.0, 1.0, -1.0, 0.5723847958486407, 0.7149219696243441, -0.34623634442129597, 0.6530563208055468, -0.627986304872186, 0.3243132873506441, -1.0, 0.40208115994581745, -0.2966387354612757, 0.019767158074273608, -1.0, 0.5770908489690131, -0.30417122031123794, -0.48640681616419745, -1.0, -0.04834559461161166, -0.5208283918190575, -0.5931563939712023, -0.5472740850131352, 0.970138011461012, -1.0, -0.7585600138061025, 0.3014411201403495, -1.0, 0.45196187135482124, -1.0, 0.6452854239000663, -1.0, 1.0, -1.0, -0.31737317002778587, -0.05837233317426745, -0.836451116911058, 1.0, -0.9422334148035697, -0.36107297420784545, -0.4593419685661435, 0.07225172542635146, 1.0, -1.0, 0.47741233422690715, 0.9434750033943367, 0.13995881627200318, -0.5520579891557081, 0.026265534920110314, 1.0, -0.6166418444779022, 0.3364133174707358, 0.8734777839389479, 0.45582218694924126, -1.0, 0.5072532480727587, 1.0, -1.0, 0.579399489690028, 0.10023148679396919, 0.21778765502178743, 1.0, 1.0, 1.0, 0.17767804244568894, -0.15677733884432932, 1.0, -0.5377839219542052, 1.0, -1.0, -0.8754554028803896, 0.6008434419628147, -0.9687564115345257, 1.0, 1.0, 0.48430979554848136, -0.875708499818095, -1.0, 0.4477821664739284, -0.35848444597920337, -1.0, 1.0, 0.8270180376874101, -1.0, -0.7851851986434667, -0.8708422675679494, -1.0, 0.750488898347355]
#     fit = my_controller.evaluate_robot(individual)
#     individual = [0.07748108901508075, 0.0543144872200746, 0.611800944572926, 1.0, -0.4232651306933256, 0.4633705420590669, 0.24573813342989603, 0.42821429281603957, 1.0, 1.0, -1.0, 0.6099860660379569, -0.027302629362711363, -0.41261168829875766, -1.0, -0.9725859383685364, -0.23299452337927706, 0.35659011708975136, 0.1449830838933087, -0.8039890939559334, 0.14290597034050295, 1.0, 0.6975723905398891, -0.16813800621070524, 1.0, 0.7571321969466553, 1.0, -0.4326151848813862, -0.39580271542183043, -1.0, 0.3517221868661927, 0.5548223214560957, 1.0, -1.0, -1.0, -0.3937271440548745, -0.99396049772706, -0.7671313612673035, -0.639623656692509, 1.0, 0.38346242144921766, 0.43003545925221726, 0.33965551911455716, -1.0, 0.5243135046952802, 0.26286158237830803, -0.36917440570932064, -1.0, 0.84560076839853, -0.6456349237705201, 0.9902152415314391, -1.0, -0.975200362901769, -1.0, -0.6955401684024795, 0.38425157381727937, 0.8650017928799958, 0.40744066591598443, 0.7246044803141072, 1.0, -0.009131542219351809, 0.6256923735742596, -1.0, 0.44720500548886993, -1.0, 1.0, -0.8501684158353914, 0.9944306605391846, -0.7616331808299206, -1.0, 1.0, -0.6755295842957614, 0.9641301045997696, 0.28131393807855676, -0.6197858247872049, -1.0, -0.2256505128554313, 0.48803924710177654, 0.25224959156876875, -1.0, -0.11859409435378594, -1.0, 0.640998719995939, -1.0, 0.40920475455447175, 0.931535061483316, -0.854237749059888, -1.0, 1.0, -0.3215059793449326, 1.0, 0.18990859072107058, -0.4520285084180654, 1.0, 0.04544261825964323, -1.0, 0.2207687796518433, 0.2675645373813212, -1.0, 0.12955101406359212, -1.0, -0.09245816427602059, 1.0, 1.0, -0.7553557147919263, 1.0, -0.6147653608786289, -1.0, 0.1577323475026284, -0.08669451727109602, 1.0, 0.937545075117995, -0.9832700045392053, 0.8357625207734796, -0.6368602318875881, -1.0, 0.5032467393557291, -0.9844009884084205, -0.7596838836271143, 1.0, -1.0, -0.034405762633260645]
#     fit = my_controller.evaluate_robot(individual)
#     individual = [0.9289540336316001, 0.21903552929840517, -1.0, -1.0, 0.6730624444463168, -0.5343068871373705, 0.004534679498982617, 0.3386671029625, -1.0, 1.0, -0.09639777596454889, -0.1705920374311611, -1.0, -1.0, -1.0, 0.4736717895282517, -1.0, 0.3088579111600518, -0.9045702868198413, -0.2897324975703127, 0.8033659612916512, 1.0, 0.523435590179999, 0.8490707003178031, -1.0, 1.0, 1.0, 0.4705016911958809, -0.3584745073993276, 1.0, -0.15752355475510657, -1.0, -0.12346028682596225, -0.03719296376032515, 0.09751764352384744, 1.0, -0.23787017309371508, -1.0, -0.07396833021698815, -0.89896999723323, 0.5389371098364627, -1.0, 0.5376666818050891, -1.0, -0.7406936074337837, -0.22399958711176635, -1.0, -0.6179050953273378, 0.40039510616902685, 0.3780004001555245, 1.0, -1.0, -0.12004315281548991, 1.0, -0.1890561381692688, -1.0, -0.8158610323540602, 1.0, 0.058461215633563636, -1.0, 0.24497560003240618, 0.7043862802702044, -0.47200871004318756, -0.16226771829660747, 1.0, -1.0, -1.0, 0.8845846955374059, 0.8849636719818894, 0.11495152662596189, 0.6422747269954064, -0.008766305272617134, -0.2325184673347717, 1.0, 0.8861465635184538, 0.3369607658021087, 1.0, -1.0, 0.21511584567803307, 0.4461353288240838, -1.0, 0.2387453587464628, 1.0, -0.722856188982131, 1.0, 0.6338503670220622, -0.0680203624376522, -0.23360007181351514, -0.42577524343414663, -1.0, -1.0, -0.5103755765055974, 0.08049415188843963, -0.8294103898439216, -1.0, -0.17751707299391736, 0.4573898253897542, -1.0, -0.17523026035156397, 0.9555520727223472, -0.9844130084132633, 0.37637803145279985, 1.0, -0.12743329546985563, -0.08704748827909832, -0.903436060212798, -0.2898361079852952, 1.0, -1.0, 1.0, -0.3268252040308846, 1.0, 0.0702091233483488, 0.8880825515198826, -1.0, 0.31950621799691764, -0.6032839923852331, 1.0, -0.4159000665798207, -0.26291006640874287, -0.7297113789291975, -0.7195276160629186]
#     fit = my_controller.evaluate_robot(individual)
#     individual=[0.12045955754904267, 0.5573580595120916, 1.0, -1.0, -1.0, -0.5578356487170485, -0.10419773237037804, 1.0, -1.0, -1.0, -0.5802273585280194, -0.367112185994471, 0.30075518954341296, -0.9863952994181852, 0.1342699415271687, -1.0, 0.1482447149079106, 0.6041507802482969, 0.10532309629778576, 0.0427527893560967, 0.5250562917659822, -1.0, 0.945998567993601, -0.2523400743931403, -0.5698411043557857, -0.1008145996343676, 0.532162275605247, 1.0, -1.0, 1.0, -1.0, 0.865951267279158, -0.4241434501446868, 0.8410704777570217, 1.0, -0.2892282675561837, 0.8980966899111988, 0.46745725168629976, -1.0, 0.1688069301902549, -0.5036028940213354, 1.0, 0.6492027832974681, -0.13225577516065656, -1.0, -0.03999753592821441, 1.0, 1.0, 0.4680947965705273, 0.09631449530060697, -1.0, -0.21694148172917865, 1.0, -0.088895850180408, 1.0, 0.7604395864475055, -1.0, -1.0, 0.9444514249460276, -0.7501568250894677, 1.0, -1.0, 1.0, 0.8849463121612116, -0.4840728333503719, -0.21519295564302138, -0.8182759812727228, -1.0, 0.6576753472535823, -0.7014602660013438, -0.924443213813222, -0.5252987224112406, -0.047348016422333626, 1.0, 1.0, 0.7522322577452565, 0.005430748759003136, -0.13385519193268997, 0.37292496386552265, -0.28614135002711955, 0.5335913248694845, 1.0, -0.997713608469123, -0.16881649543712302, -0.18100525261024974, 0.018132890557035226, 1.0, 1.0, -1.0, 0.23023051466316344, 0.9378001636521548, 0.20517630806122283, -1.0, -0.5956196317792088, -0.6301379153202203, 0.6078303610290166, -0.3868425031732712, -0.7225204654111274, 0.8714777553284293, 0.2504032217078321, -1.0, -0.5899668138565759, -0.3437941019129054, -1.0, -0.27645398997864007, 1.0, -0.11885680123696428, -1.0, -0.22161603313081682, 1.0, -0.9726925887413321, 0.49963043891739806, 0.5713457669815035, -0.4767014446041753, -1.0, 0.8871255871223186, -1.0, 1.0, -0.8170895532836263, 0.005439432928806765, 0.753889846730653, -0.7050254293435801]
#     fit = my_controller.evaluate_robot(individual)

# wireframe hard race solutions:
#[-0.5911975578945755, 0.03213236879527272, -1.0, -0.5168103993711521, -1.0, -0.2638931773931931, 0.09661932365545614, -0.9076019353230911, -0.09153913971347732, -0.22863884644663188, 1.0, 0.21346680034126356, -0.6294595762122613, -0.23304852302852747, 0.8485717619038269, 0.16037048381109922, -0.5644529328464359, -1.0, 0.27320843893828095, 0.1793708859176341, 0.0037833184428766957, -0.09417669852964948, 0.5105929483336764, -0.5421213969122375, -1.0, 1.0, -0.06490307305019101, -0.23925311291180143, -0.6754826156237604, 0.10501484185440281, -0.14525883766906342, -0.8455908873136534, 1.0, 1.0, -0.8827837516386875, 0.24559485390879968, 1.0, -0.32441790482957916, 1.0, 1.0, 0.7658818296848832, 0.27989332019717716, -0.6750350692703461, 0.28953027387903074, -1.0, -0.515722108461783, -1.0, 0.3679968642617099, 0.21895966356653737, -1.0, -1.0, -0.6513280661666465, -1.0, -0.5623157926825313, -1.0, 0.604872149961638, 1.0, -0.4360547780975465, -0.6924477553084741, -0.26716853786039385, -1.0, 0.931100706855134, 1.0, 0.4781779406817181, -1.0, 1.0, 0.4943642984961785, 1.0, 0.39727370295359715, -0.7321207797672052, -0.5468793659059482, -1.0, -0.29485534034269717, -0.8366088041499826, -1.0, 0.07966095919161992, 0.9967459364235989, -0.8349622514685211, -1.0, 0.28754575891392736, 1.0, -0.28838474645814594, -0.8795416499002684, 1.0, 0.06270645250533834, -0.5210790151793705, -1.0, -0.7566130242894888, -0.2396644732097799, 1.0, 0.006808376217511256, -0.32598403382221997, -1.0, 0.856018436921806, -1.0, 0.5043756546693522, 1.0, 0.7521107699266323, 0.18159552682560848, 0.31136446618637575, 0.75383876520814, -1.0, -0.845857005085462, 0.3570526083002237, 0.8164275265466744, -1.0, -0.5011988741526934, -0.38677259322075763, 0.9387890790815162, 0.13157345160395714, 0.27079320357784253, 0.6062293771986956, -0.353742336028557, -0.5813443847904255, -0.9004414497189741, -1.0, 0.8278845053330006, -1.0, 0.9536637055214346, 0.36418172081959765, 0.5153781005264771, -0.21233571693737593]

#[-1.0, 0.9654187228864365, 0.3537788657292931, -0.4488251261634338, 0.589439087050774, 0.3486759893441158, -1.0, 1.0, -1.0, 0.053283627980632284, 1.0, -1.0, -0.3806494459232131, 0.7387653238824591, 0.4136850795234682, 1.0, 0.20471348538310374, -1.0, -0.13089265687694274, -0.5344069492893576, -0.04966982374179874, 0.023722050735479793, -0.971563884090889, -0.07735989382346313, -0.8340312149338583, 0.5324697291545794, 1.0, 0.12733898662138615, -0.6844546045655868, 0.8101359741155213, -0.3635314024818543, 0.40313418293251246, -0.02670823180390473, 0.5203638880409831, -1.0, -0.5767242726785757, -0.5792423073550131, 0.6611162929508452, 1.0, -0.36728148849302666, 0.10109505564325595, -1.0, -0.21085582197722616, -0.26318147855296353, -0.5056881399679355, -0.21071385258278447, -1.0, 0.6995017160595791, 1.0, 1.0, -0.8008857493981667, -0.21705259871400626, 1.0, -0.28400465173611494, 0.16951888644849122, 0.23187745085646894, -0.2854848619512148, 0.32162328778036975, -0.14752998644243498, 0.12333976849755636, 0.3063660545006665, 1.0, -1.0, -0.15817580615646634, 0.6500263219186426, -0.6223283570654503, -0.19676114833072936, -0.43721137945573907, -0.580872763757168, 0.9399576797438293, 0.03498747356246823, 1.0, 0.9471399762254875, 0.4271771761778728, 1.0, 0.2929429015834379, 0.15994616605477394, -1.0, -0.5826712377548088, 0.8721506519980475, -0.9193849388061732, -1.0, -0.44355756293881327, -0.018467887407121148, 0.11629247139231708, -1.0, 1.0, -0.24291844034927537, 0.8998180708249189, 0.19038904480919172, 0.0870556616695681, 1.0, 0.6966239498582651, 1.0, -0.8775478852759387, -0.052070910236327844, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.19558566238215408, 0.2882495417934497, 0.5245601280105692, -0.9302361527834342, 1.0, -0.1636509276922709, -1.0, 0.26678246982048476, 0.23767866480892047, -1.0, 0.16273779841716254, -0.11460161256086378, 0.5764847292395262, 0.015240378881335526, -0.4766553165384359, -0.8053769251957693, -0.5625697548884515, -0.9380357772104435, 1.0, -0.26097417853720073]
#     
    #individual = [-0.5911975578945755, 0.03213236879527272, -1.0, -0.5168103993711521, -1.0, -0.2638931773931931, 0.09661932365545614, -0.9076019353230911, -0.09153913971347732, -0.22863884644663188, 1.0, 0.21346680034126356, -0.6294595762122613, -0.23304852302852747, 0.8485717619038269, 0.16037048381109922, -0.5644529328464359, -1.0, 0.27320843893828095, 0.1793708859176341, 0.0037833184428766957, -0.09417669852964948, 0.5105929483336764, -0.5421213969122375, -1.0, 1.0, -0.06490307305019101, -0.23925311291180143, -0.6754826156237604, 0.10501484185440281, -0.14525883766906342, -0.8455908873136534, 1.0, 1.0, -0.8827837516386875, 0.24559485390879968, 1.0, -0.32441790482957916, 1.0, 1.0, 0.7658818296848832, 0.27989332019717716, -0.6750350692703461, 0.28953027387903074, -1.0, -0.515722108461783, -1.0, 0.3679968642617099, 0.21895966356653737, -1.0, -1.0, -0.6513280661666465, -1.0, -0.5623157926825313, -1.0, 0.604872149961638, 1.0, -0.4360547780975465, -0.6924477553084741, -0.26716853786039385, -1.0, 0.931100706855134, 1.0, 0.4781779406817181, -1.0, 1.0, 0.4943642984961785, 1.0, 0.39727370295359715, -0.7321207797672052, -0.5468793659059482, -1.0, -0.29485534034269717, -0.8366088041499826, -1.0, 0.07966095919161992, 0.9967459364235989, -0.8349622514685211, -1.0, 0.28754575891392736, 1.0, -0.28838474645814594, -0.8795416499002684, 1.0, 0.06270645250533834, -0.5210790151793705, -1.0, -0.7566130242894888, -0.2396644732097799, 1.0, 0.006808376217511256, -0.32598403382221997, -1.0, 0.856018436921806, -1.0, 0.5043756546693522, 1.0, 0.7521107699266323, 0.18159552682560848, 0.31136446618637575, 0.75383876520814, -1.0, -0.845857005085462, 0.3570526083002237, 0.8164275265466744, -1.0, -0.5011988741526934, -0.38677259322075763, 0.9387890790815162, 0.13157345160395714, 0.27079320357784253, 0.6062293771986956, -0.353742336028557, -0.5813443847904255, -0.9004414497189741, -1.0, 0.8278845053330006, -1.0, 0.9536637055214346, 0.36418172081959765, 0.5153781005264771, -0.21233571693737593]
    #fit = my_controller.evaluate_robot(individual)

main()
