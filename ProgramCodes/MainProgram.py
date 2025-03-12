# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:19:12 2025

@author: cebrail
"""

import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from PopulationNSGA import PopulationNSGA
from Candidate import Candidate


populationSize = 20
MuCoefficient = 20
mumCoefficient = 20
crossoverProbability = .9
mutationProbability = .1
max_iteration = 1

lower = np.array([0, 0])
upper = np.array([1, 1])
csv_file = ".\\DataSet\\Covid.csv" 

df = pd.read_csv(csv_file, sep=';', lineterminator='\r')
df = df.dropna()
X = df.iloc[:, :-1]  # 🔹 All columns except the last column (independent variables)
y = df.iloc[:, -1]   # 🔹 Last column target variable
variableCount = df.shape[1] - 1
threshold = round(((variableCount * 45) / 100), 2)
k = 5  # 🔹 You can adjust K as needed
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracy_list = []

def PerformOperation(classIndex, pop, method, Dataset, fold, test):
    newPopulation = []  # For generation update

    pop.checkDominance()


    for itr in range(max_iteration):
        print('oooooooooooooooo ITERATION=', itr, ' ooooooooooooooooooooo')

        pop.split_fronts()
        if(method):
           pop.CalculateCrowdingDistanceCustom()
        else:
           pop.CalculateCrowdingDistance()
        needed = pop.IsSpecialSelectionNeeded ()
        newPopulation.clear()
        newPopulation.extend(pop.createNewGeneration(needed))
        pop.updatePopulation(newPopulation)
        pop.createParentPool()
        pop.rungaOperations(Dataset, Classes, threshold, classIndex)
        pop.checkDominance()
      
    
    if(test == 1):
        return pop.printBestSolutions(classIndex, fold)

pop = PopulationNSGA(populationSize, MuCoefficient, mumCoefficient, crossoverProbability, mutationProbability)

########################################### For objectives 3 and 4 NSGA2

for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):  # For fold number
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train = X_train.reset_index(drop=True)
    Classes = X_train.iloc[:, -1].reset_index(drop=True)
    for classIndex in range(2):
        print("Fold Number:", fold, "Class:", classIndex, " Training")
        for i in range(populationSize):
            candidateID = 'a' + str(i)
            found = False
            while(not found):
                a = Candidate(candidateID, variableCount, lower, upper, X_train, Classes, threshold, classIndex)
               
               
                   
                if(a.getObjectiveValues()[0] > 0 or a.getObjectiveValues()[1] > 0):
                    pop.add_to_population(a)
                    found = True
                   
      
          
        # Training phase 
        test = 0
        BestTrain = PerformOperation(classIndex, pop, True, X_train, fold, 0)
        
        print("Fold Number:", fold, "Class:", classIndex, " Test")
        
        BestTest = PerformOperation(classIndex, pop, True, X_test, fold, 1)
    
        for i in range(0, len(BestTest)):
            print("Test Precision:", np.round(BestTest[i].getObjectiveValues()[0], 3), "Recall:", np.round(BestTest[i].getObjectiveValues()[1], 3))    