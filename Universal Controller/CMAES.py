from deap import creator, base, tools, benchmarks, algorithms, cma
import numpy as np

'''implementation of a standard CMA-ES algorithm using the Deap library: https://deap.readthedocs.io/en/master/examples/cmaes.html
    additional functionality and further explanation of the steps below can be found in the documentation
    Installation: Deap Library: https://deap.readthedocs.io/en/master/installation.html
'''
class CMAES:
    
    np.random.seed()
    population = []
    def __init__(self, individual_size, eval_function, pop_size = 5, sigma = 0.3, threshold = 0.95, maximise = True):
        #set random seed as CMA module draws from this
        self.solution_size = individual_size   
        self.pop_size = pop_size
        self.maximise = maximise
        self.eval_function = eval_function
        self.sigma = sigma
        self.solution_threshold = threshold
        self.set_up()
        #MAYBE USE THIS TO FILL OUT DATA AND PASS BACK MY MYEA
        self.fullResults = []
    def set_up(self):
        try:
            if self.maximise:
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            else:
                creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
        except:
            pass
        
        self.toolbox = base.Toolbox()
        self.logbook = tools.Logbook()
        self.toolbox.register("evaluate", self.eval_function)
        
        #define strategy the algorithm will use, along with the population generation and update methods
        self.strategy = cma.Strategy(centroid=[0]*self.solution_size, sigma=self.sigma, lambda_= self.pop_size)
        self.toolbox.register("generate", self.strategy.generate, creator.Individual)
        self.toolbox.register("update", self.strategy.update)
        
        #establish stats to be recorded - default standard stats are set below
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        self.hof = tools.HallOfFame(1)

    def refresh(self):
        self.set_up()
        
    def run_algorithm(self, generations):
        np.random.seed() # reset seed
        self.refresh()
        print("Running standard CMAES evaluation for " + str(generations) + " generations with a population size of " + str(self.pop_size))
        hof = tools.HallOfFame(1)
        gen = 0
        solution = False
        slnEval = -1 #default value 
        #pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=5, stats=stats, halloffame=hof)
        while gen < generations:
            print("Running evals for gen: " + str(gen))
            # Generate a new population
            self.population[:] = self.pop_boundaries(self.toolbox.generate())
            # Evaluate the individuals
            fitnesses = []
            
            for i in range(self.pop_size):
                #get the fitness and endpoint of each run
                fit, endpoint = self.toolbox.evaluate(self.population[i])
                fitnesses.append((fit,))
                #check if the run was successful (fitness > solution threshold)
                if fit >= self.solution_threshold:
                    slnEval = (gen*self.pop_size) + i + 1
                    print("Solution found! On eval: " + str(slnEval))
                    print(self.population[i])
                    #if so, break
                    solution = True
                    break
                
            #map fitnesses to solutions inside cma algorithm
            for ind, fit in zip(self.population, fitnesses):
                ind.fitness.values = fit
            
            #if a solution was found break algorithm loop and exit
            if solution:
                print("break triggered") 
                break

            self.update_records(gen)
            gen += 1 
        return slnEval

    def pop_boundaries(self, population):
        '''function manually ensures population members are within expected bounds (-1, 1)
        currently no dedicated way to ensure this with Deap library'''        
        for j in range(len(population)):
            for i in range(len(population[j])):
                if population[j][i] > 1.0:
                    population[j][i] = 1.0
                elif population[j][i] < -1.0:
                    population[j][i] = -1.0
        
        return population
    
    def update_records(self, gen):
        self.hof.update(self.population)
        record = self.stats.compile(self.population)
        self.toolbox.update(self.population)
        self.logbook.record(gen=gen, evals=self.pop_size, **record)
        print(self.logbook.stream)



'''TO DO GET NOVELTY WORKING'''

'''CMA-ES algorithm augmented with novelty search
This implementation uses a ratio to determine what weight to give novelty in individual fitness assessment
the novelty_ratio variable determines this weight (between 0 - 1 weighting) - (0%-100% novelty)
Further reading on Novelty Search: https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf
'''
class NCMAES(CMAES):
    np.random.seed()    
    def __init__(self, individual_size, eval_function, pop_size = 5, novelty_ratio = 1.0, sigma = 0.3, threshold = 0.95, maximise = True):
        self.solution_size = individual_size   
        self.pop_size = pop_size
        self.maximise = maximise
        self.eval_function = eval_function
        self.sigma = sigma
        self.solution_threshold = threshold
        #novelty archive that holds a list of past solution endpoints for using to help calculate novelty of future solutions
        self.archive = []
        #novelty is calculated using the K nearest neighbours in the current population and the archive
        self.K = 15
        #novelty archive is updated with solutions that achieve a novelty score above the threshold
        self.add_threshold = 0.9
        #novelty archive also employed stochastic selection to ensure random members are also added to archive
        self.add_chance = 0.4
        #ratio determines what weight to give novelty in combined fitness formula
        self.novelty_ratio = novelty_ratio

        self.set_up()
    
    def get_n_fitnesses(self, fitnesses, endPoints):
        #additional function to calculate the individual novelty scores of each population member
        #store the list of endpoints in a class wide variable 
        self.temp_pop_endpoints = endPoints
        novelty_scores = []
        
        #get each individual novelty score
        for i in range(len(endPoints)):
            novelty_scores.append(self.calculate_novelty(endPoints[i]))
        
        #reset temp_pop_endpoints
        self.temp_pop_endpoints[:] = []
        
        nFits = []
        #function that calculates the final novelty score based on the ratio
        for fit, novelty in zip(fitnesses, novelty_scores):
            #in case of 100% novelty ratio(1.0) final score = temp score
            nFits.append((self.novelty_ratio * novelty) + (1 - self.novelty_ratio) * fit)
            
        return nFits
    
    def calculate_novelty(self, endpoint):
        #calculate novelty for an individual in the population
        temp_distances = []
        #check how close endpoint is to the other in its generation
        for i in range(len(self.temp_pop_endpoints)):
            temp = np.linalg.norm(np.array(endpoint) - np.array(self.temp_pop_endpoints[i]))
            temp_distances.append(temp)
        
        #if archive has any members, check how close enpoint is to archive members also
        if len(self.archive) > 0:  
            for i in range(len(self.archive)):
                temp_distances.append(np.linalg.norm(np.array(endpoint) - np.array(self.archive[i])))
        
        #find K closest ones
        closest = np.sort(temp_distances)[:self.K]
        #use to calculate the novelty score
        novelty = np.sum(closest)/self.K
        
        #add to archive check
        if novelty > self.add_threshold or np.random.random() < self.add_chance:
            self.archive.append(endpoint)
            
        return novelty
    
    def run_algorithm(self, generations):
        np.random.seed() # reset seed
        self.refresh()
        self.archive[:] = []
        print("Running Novelty CMAES evaluation for " + str(generations) + " generations with a population size of " + str(self.pop_size))
        hof = tools.HallOfFame(1)
        gen = 0
        solution = False
        slnEval = -1
        #pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=5, stats=stats, halloffame=hof)
        while gen < generations:
            print("Running evals for gen: " + str(gen))
            # Generate a new population
            self.population[:] = self.pop_boundaries(self.toolbox.generate())
            # Evaluate the individuals
            fitnesses = []
            endpoints = []
            for i in range(self.pop_size):
                fit, endpoint = self.toolbox.evaluate(self.population[i]) #potential issue here
                #print("rfit = " + str(fit))
                endpoints.append(endpoint)
                fitnesses.append(fit)
                if fit >= self.solution_threshold:
                    slnEval = (gen*self.pop_size) + i + 1
                    print("Solution found! On eval: " + str(slnEval)) #calculate current evaluation count
                    print(self.population[i])
                    solution = True
                    break
            n_fitnesses = self.get_n_fitnesses(fitnesses, endpoints)
            #print("Combined fitnesses for gen " + str(gen) + ":")
            #print(n_fitnesses)
            
            for ind, fit in zip(self.population, n_fitnesses):
                fitness = []
                fitness.append(fit)
                ind.fitness.values = fitness
                
            if solution:
                    print("break triggered")
                    break
                
            self.update_records(gen)
            gen +=1 
        return slnEval
