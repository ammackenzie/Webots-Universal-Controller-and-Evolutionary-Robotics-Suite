'''
Created on 12 Jul 2020

@author: Andre
'''
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import datetime
from scipy.stats import sem
from scipy.stats import t
from collections import Counter

'''Deap library is capable of storing the necessary information in a logbook - see documentation: https://deap.readthedocs.io/en/master/api/algo.html
which is fully storable via pickle, but a custom DataReporting class was chosen for full flexibility extensibility for future modification and application

DataReporting class can be used via the UniversalController  to run algorithm or hyperparameter(sigma testing for CMA-ES) testing, and will automatically save run results in a datatime and parameter stamped file
DataRerportng class can also be used to load and visualise run data via box plots - instructions in DataVisualisationandTesting.py
'''
class DataReporting:
    
    fullResults = []
    data = []
    maxEvals = 100 
    def __init__(self, saveFileName = "undefined"):
        self.saveFileName = saveFileName    
        self.loadFileName = saveFileName
        self.sigmaRuns = 10
        self.sigmaIncrements = 0.1
        self.sigmaGenerations = 5
    
    def saveTestResults(self):
        #save fullResults in a time and parameters stamped text file via pickle
        parameters = " - " + str(self.maxEvals) + "ME, " + str(self.popSize) + "PS, " + str(self.totalRuns) + "TR, "
        rawDT = datetime.datetime.now()
        date_time = rawDT.strftime("%m-%d-%Y, %H-%M-%S") #make a filename compatible datetime string
        date_time = parameters + " " + date_time + ".txt"
        date_time = self.saveFileName + date_time
        file = open(date_time,'wb')
        pickle.dump(self.fullResults, file)
        file.close()
        
        print(self.fullResults)
    
    def saveSigmaTestResults(self):
        #save fullResults in a timestamped text file via pickle
        parameters = " - " + str(self.maxEvals) + "ME, " + str(self.popSize) + "PS, " + str(self.sigmaRuns) + "SR, " + str(self.lowerLimit) + "-" + str(self.upperLimit)
        rawDT = datetime.datetime.now()
        date_time = rawDT.strftime("%m-%d-%Y, %H-%M-%S") #make a filename compatible datetime string
        date_time = parameters + ", " + date_time + ".txt"
        date_time = self.saveFileName + date_time
        
        
        file = open(date_time,'wb')
        pickle.dump(self.fullResults, file)
        file.close()
        
        print(self.fullResults)
    
        
    def sigmaTest(self, myEA, upperLimit, lowerLimit):
        #standard exmaple is for each 0.1 increment in sigma range 0.1 - 1.0
        self.maxEvals = self.sigmaGenerations * myEA.popSize # update max evals count (num of evals given to each sigma increment during each run)
        self.popSize = myEA.popSize
        self.upperLimit = upperLimit
        self.lowerLimit = lowerLimit
        for sig in (round(i * self.sigmaIncrements, 1) for i in range(round(self.lowerLimit*10), round(self.upperLimit*10)+1)):
            for reps in range(self.sigmaRuns):
                print("DataVisualisationandTesting for Sigma: " + str(sig) + ", current run: " + str(reps + 1))
                self.fullResults.append((sig, myEA.runAlgorithm(self.sigmaGenerations)))
        
        self.saveSigmaTestResults()
        
    def algorithmTest(self, myEA, generations, totalRuns):
        self.generations = generations
        self.totalRuns = totalRuns
        
        for run in range(totalRuns):
            print("Run: " + str(run))
            self.fullResults.append(myEA.runAlgorithm(self.generations))
        self.saveTestResults()
        
    def loadData(self, loadFileName = "default"):
        if loadFileName == "default":
            file = open(self.loadFileName, 'rb')
        else:
            self.loadFileName = loadFileName
            file = open(loadFileName,'rb')

        self.data.append(pickle.load(file))
    
    def displayLineGraph(self):
        graphData = []
        successes = []
        averages = []
        stds = []
        mins = []
        maxs = []
        sigmas = []
        index = 0
        
        try:
            #for each sigma and result tuple in self.data, creates a list that is as long as the count of successful runs 
            #that holds the associated evaluation count of that run
            for sig in (round(i * self.sigmaIncrements, 1) for i in range(round(self.lowerLimit*10), round(self.upperLimit*10)+1)):
                graphData.append(np.array([element[1] for element in self.data if element[0] == sig and element[1] > -1]))
                successes.append(len(graphData[index]))
                averages.append(np.average(graphData[index]))
                stds.append(np.std(graphData[index]))
                mins.append(np.min(graphData[index]))
                maxs.append(np.max(graphData[index]))
                sigmas.append(sig)
                index += 1

            
            fig, ax1 = plt.subplots()
            line1 = ax1.plot(sigmas, successes, "b-", label="Successes", color="r")
            line2 = ax1.plot(sigmas, averages, "b-", label="Average Generations", color="b")
            line3 = ax1.plot(sigmas, stds, "b-", label="Standard Deviation", color="g")
            line4 = ax1.plot(sigmas, maxs, "b-", label="Max Gens", color="purple")
            line5 = ax1.plot(sigmas, mins, "b-", label="Min Gens", color="y")
            
            #line2 = ax1.plot(sigmas, fit_min, "b-", label="min Fitness", color="b")
            #line3 = ax1.plot(sigmas, fit_avg, "b-", label="avg Fitness", color="g")
            ax1.set_xlabel("Sigma")
            ax1.set_ylabel("", color="b")
            for tl in ax1.get_yticklabels():
                tl.set_color("b")
              
                  
            lns = line1 +line2 +line3 + line4 + line5
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc="center right")
            plt.yticks(np.arange(0, max(maxs), 5))
            plt.xticks(np.arange(min(sigmas), max(sigmas), 0.1))
            plt.show()
        except:
            print("lowerLimit and/or upperLimit values do not match loaded data")
    
    def displayAlgorithmBoxPlots(self, algorithmOne = "default", algorithmTwo = "default", algorithmThree = "default", algorithmFour = "default"):
        data = []
        xAxisData = [algorithmOne, algorithmTwo, algorithmThree, algorithmFour]
        xAxisLabel = "Algorithms"
        
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j] == -1: # default value - no solution was found on this run
                    self.data[i][j] = self.maxEvals
            data.append(self.data[i])
            
        title = "CMA-ES DataVisualisationandTesting on MiddleWall"
        
        self.generateBoxPlots(data, xAxisData, xAxisLabel, 10, title)
        
    def displaySigmaBoxPlots(self):
        data = []
        xAxisData = []
        index = 0
        print(self.data)
        #for each sigma and result tuple in self.data, creates a list that is as long as the count of successful runs 
        #that holds the associated evaluation count of that run
        for sig in (round(i * self.sigmaIncrements, 1) for i in range(round(self.lowerLimit*10), round(self.upperLimit*10)+1)):
            data.append(np.array([element[1] for element in self.data[0] if element[0] == sig]))

            for j in range(len(data[index])):
                if data[index][j] == -1:
                    data[index][j] = self.maxEvals
            xAxisData.append(sig)
            index += 1

        if self.loadFileName == "undefined":
            title = self.loadFileName
        else:
            splitTitle = self.loadFileName.split(" - ", 2)
            title = splitTitle[0] + ", Environment: " + splitTitle[1]
        self.pandTtest(data)
        self.generateBoxPlots(data, xAxisData, "Sigmas", 10, title)
        
    
    def generateBoxPlots(self, data, xAxisData, xAxisLabel, yTicks, title):
        '''reference: Matplotlib documentation, https://matplotlib.org/3.1.1/gallery/statistics/boxplot_demo.html'''
        fig, ax1 = plt.subplots(figsize=(10, 6))
        #fig.canvas.set_window_title('Sigma DataVisualisationandTesting')
        fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
        
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
        
        ax1.set_title(title)
        ax1.set_xlabel(xAxisLabel)
        
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
        top = self.maxEvals + 10
        bottom = 0
        ax1.set_ylim(bottom, top)
        
        ax1.set_xticklabels(xAxisData,
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
        
        # Finally, add a basic legend
        # fig.text(0.80, 0.08, f'{N} Random Numbers',
        #          backgroundcolor=box_colors[0], color='black', weight='roman',
        #          size='x-small')
        # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
        #          backgroundcolor=box_colors[1],
        #          color='white', weight='roman', size='x-small')
        fig.text(0.80, 0.148, '+', color='black', backgroundcolor='gray',
                 weight='roman', size='medium')
        fig.text(0.815, 0.15, ' Outliers', color='black', weight='roman',
                  size='medium')
        
        plt.yticks(np.arange(0, self.maxEvals, yTicks))
        plt.show()
        #except:
            #print("Error") #'''TO CHANGE '''
    
    def independentTTest(self, data1, data2, alpha):
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
    
    def pandTtest(self, sigmaData):
        alpha = 0.05
        #t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
        count = 0
        rejections = 0
        significantSigmas = []
        mostSignificant = []

        for j in range(len(sigmaData)):
            for i in range(len(sigmaData)):
                if i <= j:
                    pass
                else:
                    count += 1
                    print("For sigma " + str((j+1)/10) + " and " + str((i+1)/10))
                    if self.independentTTest(sigmaData[j], sigmaData[i], alpha):
                        rejections += 1
                        significantSigmas.append(((j+1)/10, (i+1)/10))
                        mostSignificant.append((j+1)/10)
                        mostSignificant.append((i+1)/10)
                    
        print("total comparisons: " + str(count))
        print("Total rejections: " + str(rejections))
        print(significantSigmas)
        #biggest = np.max([mostSignificant.count(element) for element in mostSignificant])
        countPairs = Counter(mostSignificant)
        print(countPairs)


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
        