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

At the end of each run, a datetime stamped map is saved with the save_file variable as a name which can be loaded and used in future runs
A separate file will also be saved/updated with a data log detailing the run details and parameters to help keep track of the maps development
'''


class MAPElites():
    np.random.seed()
    solution_threshold = 0.90
    loaded_state = False
    '''The below maxVeloity and maxAngularVelocity applies to the e-puck robot and will need to be manually changed for different morphologies'''
    '''Limitation of Webots - can't directly access what the maximum angular velocity is for a given robot - has to be determined manually through testing in simulation '''
    
    #maxDFS in easyrace = 1.23138
    #maxdfs in multimaze = 0.78749
    #maxdfs in middlewall = 0.8942810
    #maxdfs in escape room 0.63008 (already accounted for)
    
    heatmap_title = "undefined"
    y_axis_label = "Average Velocity (m/s)"
    
    g_objective_fitnesses = []
    g_behavioural_descriptors = []
    
    def __init__(self, individual_size, eval_function, bin_count = 10, tournament_size = 5, max_velocity = 0.125, max_distance = 1.2, x_values=[-0.465, 0.465], y_values=[-0.465, 0.465]):
        self.bins = bin_count
        self.eval_function = eval_function
        #max observed velocity -.125 m/s
        self.velocity = [round(e * (max_velocity/self.bins), 3) for e in range(1, (self.bins + 1))]
        self.x_values_bins = [(x_values[0] + round(e * ((x_values[1] - x_values[0])/self.bins), 3)) for e in range(1, (self.bins + 1))]
        self.y_values_bins = [(y_values[0] + round(e * ((y_values[1] - y_values[0])/self.bins), 3)) for e in range(1, (self.bins + 1))]
        #max observed angular velocity 2.1 rad/s
        #self.distance_from_start = [round(e * (max_distance/self.bins), 3) for e in range(1, (self.bins + 1))]
        self.member_size = individual_size
        self.mutation_rate = 0.1
        self.tournament_size = tournament_size
        self.search_space = np.zeros([self.bins,self.bins, self.bins])
        self.saved_members = np.zeros([self.bins,self.bins, self.bins,  self.member_size])
        #saves any successful solutions in format: [individual, coordinates, eval]
        self.successes = []
        self.save_file = "EMAP - Unspecified"
    
    def insert(self, behaviours, fitness, individual):
        #print(behaviours)
        c1 = np.digitize(behaviours[0], self.velocity)
        c2 = np.digitize(behaviours[1], self.x_values_bins)
        c3 = np.digitize(behaviours[2], self.y_values_bins)
        #print(self.y_values_bins)
        #print(c1)
        #print(c2)
        #print(c3)
        #finds the cell the individual belongs to, keeps if better fitness
        if self.search_space[c1][c2][c3] < fitness:
            self.search_space[c1][c2][c3] = fitness
            self.saved_members[c1][c2][c3][:] = individual
    
        return [c1, c2, c3]
    def new_member(self):
#         new_member = np.zeros(self.member_size)
#         for i in range(self.member_size):
#             new_member[i] = np.random.uniform(-1, 1)
        new_member = np.random.uniform(-1, 1, self.member_size)
        return new_member
    
    def gaussian_mutation(self, individual):
        #create boolean array determining whether to mutate a given index or not
        mutate = np.random.uniform(0, 1, individual.shape) < self.mutation_rate
   
        for i in range(len(individual)):
            if mutate[i]:
                individual[i] = np.random.normal(0, 0.5)
                if individual[i] > 1.0:
                    individual[i] = 1.0
                if individual[i] < -1.0:
                    individual[i] = -1.0
    
        return individual
    
    def check_empty_adjascent(self, member):
        #check number of empty bins next to member
        empty_count = 0
        for i in range(member[0]-1, member[0]+2):
            for j in range(member[1]-1, member[1]+2):
                for k in range(member[2]-1, member[2]+2):
                #ensure we are not checking member or index outside of search_space
                    if [i, j, k] == member or i < 0 or j < 0 or k < 0 or i >= self.bins or j >= self.bins or k >= self.bins:
                        pass
                    else:
                        if self.search_space[i][j][k] == 0:
                            empty_count += 1
                        else:
                            pass
        return empty_count
        
    def combine_loaded_map(self, filename):
        #for use when plotting combined heatmap from multiple run results
        f = open(filename,'rb')
        data = list(zip(*(pickle.load(f))))
        for i in range(self.bins):
            for j in range(self.bins):
                for k in range(self.bins):
                    if data[0][i][j][k] > self.search_space[i][j][k]:
                        self.search_space[i][j][k] = data[0][i][j][k]
                    
    def get_random_member(self):
        c1 = np.random.randint(0, self.bins)
        c2 = np.random.randint(0, self.bins)
        c3 = np.random.randint(0, self.bins)
        while np.max(self.saved_members[c1][c2][c3]) == 0 or np.min(self.saved_members[c1][c2][c3]) == 0 :
            c1 = np.random.randint(0, self.bins)
            c2 = np.random.randint(0, self.bins)
            c3 = np.random.randint(0, self.bins)
        
        return self.saved_members[c1][c2][c3]
    
    def tournament_select(self):
        #select first initial random choice and set to best choice so far
        best_member_ID = self.get_random_ID()
        most_empty_bins = self.check_empty_adjascent(best_member_ID)
        
        for round in range(self.tournament_size-1):
            #select new random member
            temp_member_ID = self.get_random_ID()
            temp_empty_bins = self.check_empty_adjascent(temp_member_ID)
            #if new choice has more empty nearby cells, set as best choice
            if temp_empty_bins > most_empty_bins:
                most_empty_bins = temp_empty_bins
                best_member_ID = temp_member_ID
        #retrieve the chosen member itself from the saved members array
        return self.saved_members[best_member_ID[0]][best_member_ID[1]][best_member_ID[2]]
    
    def get_random_ID(self):
        c1 = np.random.randint(0, self.bins)
        c2 = np.random.randint(0, self.bins)
        c3 = np.random.randint(0, self.bins)
        while np.max(self.search_space[c1][c2][c3]) == 0:
            c1 = np.random.randint(0, self.bins)
            c2 = np.random.randint(0, self.bins)
            c3 = np.random.randint(0, self.bins)
        return [c1, c2, c3]

    def save_map(self):
        rawDT = datetime.datetime.now()
        date_time = rawDT.strftime("%m-%d-%Y, %H-%M-%S") #make a filename compatible datetime string
        parameters = " " + str(self.tournament_size) + "TS, " + str(self.bins) + "BC, " + str(self.generations) + "gens" 
        self.save_file = self.save_file + parameters + date_time
        f = open(self.save_file + "FULL MAP" + ".txt",'wb')
        pickle.dump(zip(self.search_space, self.saved_members), f)
        f.close()
        self.save_log_data()
#         if len(self.successes) > 0:
#             self.saveSuccesses()
            
    def saveSuccesses(self):
        f = open(self.save_file + " SUCCESSES.txt",'wb')
        pickle.dump(self.successes, f)
        f.close()
        
    def save_log_data(self):
        dt = datetime.datetime.now()
        date_time = dt.strftime("%m/%d/%Y, %H:%M:%S")
        info_string = "Following data added on: " + date_time
        info_string += "\nBin count: " + str(self.bins)
        info_string += "\nGenerations: " + str(self.generations)
        info_string += "\nInitial Pop Size: " + str(self.initial_pop_size)
        info_string += "\nMutationRate: " + str(self.mutation_rate)
        info_string += "\nTournamentSize: " + str(self.tournament_size)
        info_string += "\nSuccesses found: " + str(len(self.successes))
        info_string += "\nSuccessful cells: " + str(self.solution_count)
        if len(self.successes) > 0:
            info_string += "\nFirst successful solution found on eval: " + str(self.successes[0][2])
        info_string += "\n____________________________________________________"
        tf = open(self.save_file + " DataLog.txt", 'w')
        tf.write(info_string)
        tf.close()
    
    def load_map(self, save_file):
        f = open(save_file,'rb')
        data = list(zip(*(pickle.load(f))))
        for i in range(self.bins):
            for j in range(self.bins):
                for k in range(self.bins):
                    self.search_space[i][j][k] = data[0][i][j][k]
                    self.saved_members[i][j][k] = data[1][i][j][k]
        self.loaded_state = True

    
    def refresh(self):
        self.search_space[:] = np.zeros([self.bins,self.bins, self.bins])
        self.saved_members[:] = np.zeros([self.bins,self.bins, self.bins, self.member_size])
        self.successes[:] = []
    #minimum effective generations is 10
    def run_algorithm(self, generations):
        self.g_behavioural_descriptors[:] = []
        self.g_objective_fitnesses[:] = []
        np.random.seed()
        print("Running MAP-Elites evaluation for " + str(generations) + " generations with a bin count of: " + str(self.bins))
        
        #use the first 10% of the total generations to generate initial cell entries or bin_count *50 - whatever smaller
        self.initial_pop_size  = round(generations/10)
        #check if map is already loaded
        if self.loaded_state:
            self.initial_pop_size = 0
        else:
            #if not its a new run so refresh class vairables
            self.refresh()
            
        self.generations = generations
        gen = 0
        while gen < generations:
            #create a new empty array of correct size
            new_member = np.zeros(self.member_size)
            #if we are still running the initial batch, randomly create a new member
            if gen < self.initial_pop_size:
                new_member[:] = self.new_member()
            else:
                #otherwise use tournament select to chose a cell to mutate
                new_member[:] = self.tournament_select()
                new_member[:] = self.gaussian_mutation(new_member)
            
            #get the behavioural description of the new member through evaluation
            average_V, endpoint, distance_FS, fitness = self.eval_function(new_member, map_elites =True)
          
            behaviours = [average_V, endpoint[0], endpoint[1]]
            self.g_behavioural_descriptors.append(behaviours)
            self.g_objective_fitnesses.append(fitness)
            ##pass to insert function which determines whether to keep or discard
            coordinates = self.insert(behaviours, fitness, new_member)
            
            #check if it's a successful solution
            if fitness > self.solution_threshold:
                self.successes.append([new_member, coordinates, gen])
                print("Solution found! On eval: " + str(gen) + ", at map coordinate: " + str(coordinates[0]) + "," + str(coordinates[1]) + "," + str(coordinates[2]))
                break
            #every 100 iterations, prints the search_space to the console for monitoring
            if gen % 100 == 0 and gen > 0:
                print(gen)
                print(self.search_space)
            gen += 1
        
        #visualise the map and the solutions found in the console
        self.visualise_map()
        self.save_map()
        return gen, self.g_behavioural_descriptors, self.g_objective_fitnesses
        
        
    def visualise_map(self):
        self.solution_count = 0
        solutions = []
        solution_space = np.zeros([self.bins,self.bins, self.bins])
        for i in range(self.bins):
            for j in range(self.bins):
                for k in range(self.bins):
                    if self.search_space[i][j][k] > self.solution_threshold:
                        self.solution_count += 1
                        solution_space[i][j][k] = 1
                        solutions.append(self.saved_members[i][j][k])
                
        
        print("total solutions found: ")
        print(self.solution_count)
        print("solution space:")
        print(solution_space)
        print(self.search_space)
        return solutions
    
    def generate_heatmap(self):
        df = pd.DataFrame(self.search_space)
        ax = sns.heatmap(df)
        ax.set_xticklabels(self.distance_from_start,
                                    rotation=0, fontsize=12)
        ax.set_yticklabels(self.velocity,rotation=0, fontsize=12)
        ax.set_xlabel("Distance from Start (m)", fontsize=14)
        ax.set_ylabel("Average Velocity (m/s)", fontsize=14)
        ax.set_title("MAP-Elites Heatmap - Easy Race - Custom e-puck - RNN", fontsize=14)
        plt.show()
