import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import datetime
from scipy.stats import sem
from scipy.stats import t
from scipy.stats import ttest_ind
from collections import Counter


'''Deap library is capable of storing the necessary information in a logbook - see documentation: https://deap.readthedocs.io/en/master/api/algo.html
which is fully storable via pickle, but a custom DataReporting class was chosen for full flexibility extensibility for future modification and application

DataReporting class can be used via the UniversalController  to run algorithm or hyperparameter(sigma testing for CMA-ES) testing, and will automatically save run results in a datatime and parameter stamped file
DataRerportng class can also be used to load and visualise run data via box plots - instructions in DataVisualisationandTesting.py
'''
class DataReporting:
    
    full_results = []
    data = []
    max_evals = 100 
    y_ticks = 10
    graph_padding = 10
    x_axis_label = ""
    y_axis_label = ""
    blue_label = ""
    tan_label = ""
    def __init__(self, save_file_name = "undefined"):
        self.save_file_name = save_file_name    
        self.load_file_name = save_file_name
        self.sigma_runs = 10
        self.sigma_increments = 0.1
        self.sigma_generations = 5
        
    def save_test_results(self):
        #save full_results in a time and parameters stamped text file via pickle
        parameters = " - " + str(self.max_evals) + "ME, " + str(self.pop_size) + "PS, " + str(self.total_runs) + "TR, "
        rawDT = datetime.datetime.now()
        date_time = rawDT.strftime("%m-%d-%Y, %H-%M-%S") #make a filename compatible datetime string
        date_time = parameters + " " + date_time + ".txt"
        date_time = self.save_file_name + date_time
        file = open(date_time,'wb')
        pickle.dump(self.full_results, file)
        file.close()
        
        print(self.full_results)
    
    def save_map_elites_test_results(self):
        #save full_results in a time and parameters stamped text file via pickle
        parameters = " " + str(self.algorithm.tournament_size) + "TS, " + str(self.algorithm.bins) + "BC, " + str(self.algorithm.generations) + "gens" 
        rawDT = datetime.datetime.now()
        date_time = rawDT.strftime("%m-%d-%Y, %H-%M-%S") #make a filename compatible datetime string
        date_time = parameters + " " + date_time + ".txt"
        date_time = self.save_file_name + date_time
        tf = open(date_time, 'w')
        tf.write(str(self.full_results))
        tf.close()
        print(self.full_results)
    
    def save_sigma_test_results(self):
        #save full_results in a timestamped text file via pickle
        parameters = " - " + str(self.max_evals) + "ME, " + str(self.pop_size) + "PS, " + str(self.sigma_runs) + "SR, " + str(self.lower_limit) + "-" + str(self.upper_limit)
        rawDT = datetime.datetime.now()
        date_time = rawDT.strftime("%m-%d-%Y, %H-%M-%S") #make a filename compatible datetime string
        date_time = parameters + ", " + date_time + ".txt"
        date_time = self.save_file_name + date_time
        
        file = open(date_time,'wb')
        pickle.dump(self.full_results, file)
        file.close()
        
        print(self.full_results)
    
        
    def sigmaTest(self, my_EA, upper_limit, lower_limit):
        #standard exmaple is for each 0.1 increment in sigma range 0.1 - 1.0
        self.max_evals = self.sigma_generations * my_EA.pop_size # update max evals count (num of evals given to each sigma increment during each run)
        self.pop_size = my_EA.pop_size
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        for sig in (round(i * self.sigma_increments, 1) for i in range(round(self.lower_limit*10), round(self.upper_limit*10)+1)):
            for reps in range(self.sigma_runs):
                print("DataVisualisationandTesting for Sigma: " + str(sig) + ", current run: " + str(reps + 1))
                self.full_results.append((sig, my_EA.run_algorithm(self.sigma_generations)))
        
        self.save_sigma_test_results()
        
    def algorithm_test(self, my_EA, generations, total_runs, map_elites = False):
        self.algorithm = my_EA
        self.generations = generations
        self.total_runs = total_runs
        if map_elites:
            self.max_evals = generations
            my_EA.save_file = self.save_file_name
        else:
            self.pop_size = my_EA.pop_size
            self.max_evals = self.generations * self.pop_size
        for run in range(total_runs):
            print("Run: " + str(run))
            self.full_results.append(my_EA.run_algorithm(self.generations))
        if map_elites:
            self.save_map_elites_test_results()
        else:
            self.save_test_results()
    
    
    def control_group_test(self, individual_size, eval_function, generations = 10000, total_runs = 1):
        self.generations = generations
        self.individual_size = individual_size
        self.total_runs = total_runs
        self.eval_func = eval_function
        self.max_evals = self.generations 
        for run in range(total_runs):
            print("Running control group test for " + str(self.generations) + " generations. Run: " + str(run))
            self.full_results.append(self.run_control_group())
        
        parameters = " - " + str(self.max_evals) + "ME, " + str(self.total_runs) + "TR, "
        rawDT = datetime.datetime.now()
        date_time = rawDT.strftime("%m-%d-%Y, %H-%M-%S") #make a filename compatible datetime string
        date_time = parameters + " " + date_time + ".txt"
        date_time = self.save_file_name + date_time
        tf = open(date_time, 'w')
        tf.write(str(self.full_results))
        tf.close()
        
        print(self.full_results)
            
         
    def run_control_group(self):
        member = np.zeros(self.individual_size)
        gen = 0
        while gen < self.generations:
            member[:] = np.random.uniform(-1, 1, self.individual_size)
            fit, endpoint = self.eval_func(member)
            if fit > 0.95: #solution threshold
                print("Solution found! On evaluation: " + str(gen))
                print(member)
                break
            gen += 1
        
        return gen
    def load_data(self, load_file_name = "default"):
        if load_file_name == "default":
            file = open(self.load_file_name, 'rb')
        else:
            self.load_file_name = load_file_name
            file = open(load_file_name,'rb')

        self.data.append(pickle.load(file))
    
    def display_line_graph(self, title = "undefined"):
        #graphData = []
        rejections = []
#         successes = []
#         averages = []
#         stds = []
#         mins = []
#         maxs = []
        sigmas = []
        index = 0
        
        try:
            #for each sigma and result tuple in self.data, creates a list that is as long as the count of successful runs 
            #that holds the associated evaluation count of that run
            for sig in (round(i * self.sigma_increments, 1) for i in range(round(self.lower_limit*10), round(self.upper_limit*10)+1)):
                #graphData.append(np.array([element[1] for element in self.data if element[0] == sig and element[1] > -1]))
                rejections.append(self.data[index])
#                 successes.append(len(graphData[index]))
#                 averages.append(np.average(graphData[index]))
#                 stds.append(np.std(graphData[index]))
#                 mins.append(np.min(graphData[index]))
#                 maxs.append(np.max(graphData[index]))
                sigmas.append(sig)
                index += 1

            
            fig, ax1 = plt.subplots()
            ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                       alpha=0.5)
            line1 = ax1.plot(sigmas, rejections, "b-", label="Percentage", color="r")
#             line2 = ax1.plot(sigmas, averages, "b-", label="Average Generations", color="b")
#             line3 = ax1.plot(sigmas, stds, "b-", label="Standard Deviation", color="g")
#             line4 = ax1.plot(sigmas, maxs, "b-", label="Max Gens", color="purple")
#             line5 = ax1.plot(sigmas, mins, "b-", label="Min Gens", color="y")
            
            #line2 = ax1.plot(sigmas, fit_min, "b-", label="min Fitness", color="b")
            #line3 = ax1.plot(sigmas, fit_avg, "b-", label="avg Fitness", color="g")
            ax1.set_title(title, fontsize=14)
        
            ax1.set_xlabel(self.x_axis_label, fontsize=14)
            ax1.set_ylabel(self.y_axis_label, fontsize=14)
            
            for tl in ax1.get_yticklabels():
                tl.set_color("b")
              
                  
            lns = line1 #+line2 +line3 + line4 + line5
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc="center right")
            #plt.y_ticks(np.arange(0, max(maxs), 5))
            plt.yticks(np.arange(0, self.max_evals, self.y_ticks))
            plt.xticks(np.arange(min(sigmas), max(sigmas)+0.1, 0.1))
            plt.show()
        except:
            print("lower_limit and/or upper_limit values do not match loaded data")
    
    def displayAlgorithmBoxPlots(self, algorithms, title = "unspecififed"): #algorithmOne = "default", algorithmTwo = "default", algorithmThree = "default", algorithmFive = "default", algorithmThree = "default", algorithmFour = "default"):
        data = []
        x_axis_data = []
        for algorithm in algorithms:
            x_axis_data.append(algorithm)
        
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j] == -1: # default value - no solution was found on this run
                    self.data[i][j] = self.max_evals
            data.append(self.data[i])
        
        self.generateBoxPlots(data, x_axis_data, title)
        
    def displaySigmaBoxPlots(self, title = "undefined"):
        data = []
        x_axis_data = []
        index = 0
        print(self.data)
        #for each sigma and result tuple in self.data, creates a list that is as long as the count of successful runs 
        #that holds the associated evaluation count of that run
        for sig in (round(i * self.sigma_increments, 1) for i in range(round(self.lower_limit*10), round(self.upper_limit*10)+1)):
            data.append(np.array([element[1] for element in self.data[0] if element[0] == sig]))

            for j in range(len(data[index])):
                if data[index][j] == -1:
                    data[index][j] = self.max_evals
            x_axis_data.append(sig)
            index += 1

        #if self.load_file_name == "undefined":
        #    title = self.load_file_name
        #else:
        #    splitTitle = self.load_file_name.split(" - ", 2)
        #    title = splitTitle[0] + ", Environment: " + splitTitle[1]
        #self.p_and_t_test(data)

        self.sigma_significance_testing(data)
        #self.basic_t_test(array1, array2, alpha)
        self.generateBoxPlots(data, x_axis_data, title)
        
    
    def generateBoxPlots(self, data, x_axis_data, title):
        '''reference: Matplotlib documentation, https://matplotlib.org/3.1.1/gallery/statistics/boxplot_demo.html'''
        fig, ax1 = plt.subplots(figsize=(10, 6))
        #fig.canvas.set_window_title('Sigma DataVisualisationandTesting')
        #fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
        
        bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='red', marker='+')
        
        # Add a horizontal grid to the plot, but make it very light in color
        # so we can use it for reading data values but not be distracting
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                       alpha=0.5)
        
        # Hide these grid behind plot objects
        ax1.set_axisbelow(True)
        
        ax1.set_title(title, fontsize=14)
        
        ax1.set_xlabel(self.x_axis_label, fontsize=14)
        ax1.set_ylabel(self.y_axis_label, fontsize=14)
        ax1.set_ylim()
        
        # Now fill the boxes with desired colors
        #box_colors = ['darkkhaki', 'royalblue']
        box_colors = ['royalblue', 'tan']
        num_boxes = len(data)
        medians = np.empty(num_boxes)
        for i in range(num_boxes):
            box = bp['boxes'][i]
            boxX = []
            boxY = []
            for j in range(5):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            box_coords = np.column_stack([boxX, boxY])
            # Alternate between Dark Khaki and Royal Blue
            ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
            # Now draw the median lines back over what we just filled in
            med = bp['medians'][i]
            medianX = []
            medianY = []
            for j in range(2):
                medianX.append(med.get_xdata()[j])
                medianY.append(med.get_ydata()[j])
                ax1.plot(medianX, medianY, 'k')
            medians[i] = medianY[0]
        
        # Set the axes ranges and axes labels
        ax1.set_xlim(0.5, num_boxes + 0.5)
        top = self.max_evals + self.graph_padding
        bottom = 0
        ax1.set_ylim(bottom, top)
        
        ax1.set_xticklabels(x_axis_data,
                            rotation=0, fontsize=12)
        
        # Due to the Y-axis scale being different across samples, it can be
        # hard to compare differences in medians across the samples. Add upper
        # X-axis tick labels with the sample medians to aid in comparison
        # (just use two decimal places of precision)
        pos = np.arange(num_boxes) + 1
        upper_labels = [str(np.round(s, 2)) for s in medians]
        weights = ['bold', 'semibold']
        for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
            k = tick % 2
            ax1.text(pos[tick], .95, upper_labels[tick],
                     transform=ax1.get_xaxis_transform(),
                     horizontalalignment='center', size='x-small',
                     weight=weights[k], color=box_colors[k])
        
        
        #Finally, add a basic legend f'{N} Random Numbers'
        fig.text(0.80, 0.01, self.tan_label,
                 backgroundcolor=box_colors[1],
                 color='black', weight='roman', size='small')
        fig.text(0.80, 0.04, self.blue_label,
                 backgroundcolor=box_colors[0], color='black', weight='roman',
                 size='small')
        
#         fig.text(0.80, 0.148, '+', color='black', backgroundcolor='gray',
#                  weight='roman', size='medium')
#         fig.text(0.815, 0.15, ' Outliers', color='black', weight='roman',
#                   size='medium')
        
        plt.yticks(np.arange(0, self.max_evals + 10, self.y_ticks))
        plt.show()
        #except:
            #print("Error") #'''TO CHANGE '''
    
    def sigma_significance_testing(self, sigmaData):
        
        count = 0
        rejections = 0
        results = np.zeros((len(sigmaData), len(sigmaData)))
        for j in range(len(sigmaData)):
            for i in range(len(sigmaData)):
                if i <= j:
                    pass
                else:
                    count += 1
                    print("For sigma " + str((j+1)/10) + " and " + str((i+1)/10))
                    if self.basic_t_test(sigmaData[j], sigmaData[i]):
                        results[j][i] = -(i+1)/10
                        rejections += 1
                        #significant_sigmas.append(((j+1)/10, (i+1)/10))
                        #most_significant.append((j+1)/10)
                        #most_significant.append((i+1)/10)
                    else:
                        results[j][i] = 1
            results[j][0] = (j+1)/10
        print("Rejections: " + str(rejections))
        print(results)
    
    def basic_t_test(self, array1, array2, alpha = 0.05):
        #assuming both arrays are of equal size
        stats, p = ttest_ind(array1, array2)
        
        if p > alpha:
            print("Fails to reject")
            return False
        else:
            print("Rejects NULL HYPOTHESIS") 
            return True   
        
        
    
    def independent_t_test(self, data1, data2, alpha):
        '''Reference: 
        BrownLee, Jason, 2019, 'How to Code the Student's t-Test from Scratch in Python', MachineLeanring Mastery, retrieved from: https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
        '''
        rejected = False
        tReject = False
        pReject = False
        # calculate means
        mean1, mean2 = np.mean(data1), np.mean(data2)
        # calculate standard errors
        se1, se2 = sem(data1), sem(data2)
        # standard error on the difference between the samples
        sed = np.sqrt(se1**2.0 + se2**2.0)
        # calculate the t statistic
        t_stat = (mean1 - mean2) / sed
        # degrees of freedom
        df = len(data1) + len(data2) - 2
        # calculate the critical value
        cv = t.ppf(1.0 - alpha, df)
        # calculate the p-value
        p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
        # return everything
        print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
        # interpret via critical value
        if abs(t_stat) <= cv:
            print('Accept null hypothesis that the means are equal.')
        else:
            print('Reject the null hypothesis that the means are equal.')
            tReject = True
        # interpret via p-value
        if p > alpha:
            print('Accept null hypothesis that the means are equal.')
        else:
            print('Reject the null hypothesis that the means are equal.')
            pReject = True
        
        rejected = tReject and pReject
        #return t_stat, df, cv, p
        return rejected
    
    def p_and_t_test(self, sigmaData):
        alpha = 0.05
        #t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
        count = 0
        rejections = 0
        significant_sigmas = []
        most_significant = []

        for j in range(len(sigmaData)):
            for i in range(len(sigmaData)):
                if i <= j:
                    pass
                else:
                    count += 1
                    print("For sigma " + str((j+1)/10) + " and " + str((i+1)/10))
                    if self.independent_t_test(sigmaData[j], sigmaData[i], alpha):
                        rejections += 1
                        significant_sigmas.append(((j+1)/10, (i+1)/10))
                        most_significant.append((j+1)/10)
                        most_significant.append((i+1)/10)
                    
        print("total comparisons: " + str(count))
        print("Total rejections: " + str(rejections))
        print(significant_sigmas)
        #biggest = np.max([most_significant.count(element) for element in most_significant])
        count_pairs = Counter(most_significant)
        print(count_pairs)

    #WIP below
    def pvalue_101(self, mu, sigma, samp_size, samp_mean=0, deltam=0):
        np.random.seed(1234)
        s1 = np.random.normal(mu, sigma, samp_size)
        if samp_mean > 0:
            print(len(s1[s1>samp_mean]))
            outliers = float(len(s1[s1>samp_mean])*100)/float(len(s1))
            print('Percentage of numbers larger than {} is {}%'.format(samp_mean, outliers))
        if deltam == 0:
            deltam = abs(mu-samp_mean)
        if deltam > 0 :
            outliers = (float(len(s1[s1>(mu+deltam)]))
                        +float(len(s1[s1<(mu-deltam)])))*100.0/float(len(s1))
            print('Percentage of numbers further than the population mean of {} by +/-{} is {}%'.format(mu, deltam, outliers))
    
        fig, ax = plt.subplots(figsize=(8,8))
        fig.suptitle('Normal Distribution: population_mean={}'.format(mu) )
        plt.hist(s1)
        plt.axvline(x=mu+deltam, color='red')
        plt.axvline(x=mu-deltam, color='green')
        plt.show()
        