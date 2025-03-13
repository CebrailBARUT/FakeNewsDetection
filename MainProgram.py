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
import Purposes as objectives
from os import path
import matplotlib.pyplot as plt
import csv


populationSize = 20
MuCoefficient = 20
mumCoefficient = 20
crossoverProbability = .9
mutationProbability = .1
max_iteration = 100
numOfObjective=2
ObjectiveFunciton=["precision","recall"]
lower = np.array([0, 0])
upper = np.array([1, 1])
csv_file = ".\\DataSet\\Syrian.csv" 
test_file = "Experiments/Test/Test.csv"
df = pd.read_csv(csv_file, sep=';', lineterminator='\r')
df = df.dropna()
X = df.iloc[:, :-1]  # 🔹 All columns except the last column (independent variables)
y = df.iloc[:, -1]   # 🔹 Last column target variable
variableCount = df.shape[1] - 1
threshold = round(((variableCount * 45) / 100), 2)
k = 5  # 🔹 You can adjust K as needed
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracy_list = []
def file_exists(file):
    if not path.exists(file):
        with open(file, mode='a', newline='') as file_writer:
            header = ["Fold", "Class", "Precision", "Recall"]
            writer = csv.writer(file_writer)
            writer.writerow(header)
def append_to_csv(file, element):
    with open(file, 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(element)

def findSimilarity(datainDataSet, datainPopulation):
    
    return np.sum(datainDataSet == datainPopulation)
def calculateValues(result,result_class, dataset, classes):
    
    tp=0
    tn=0
    fp=0
    fn=0
    for j in range(len(dataset)):
       
        datainDataSet = np.array(dataset.iloc[j].values.tolist(), int)
        similarity_score = findSimilarity(datainDataSet,result)
        
        if similarity_score > threshold and classes[j] == result_class:
            tp+=1
           
        elif similarity_score > threshold and classes[j] != result_class:
            fp+=1
            
        elif similarity_score < threshold and classes[j] == result_class:
           fn+=1
            
        else:
            tn+=1
          
    return [tp,tn,fp,fn]
   
def calculateObjectiveValues(tptnfpfn):
   
    values = np.zeros(numOfObjective)
    for i in range(numOfObjective):
        func = getattr(objectives, ObjectiveFunciton[i])
        
        values[i] = func(tptnfpfn)
    
    
    return values
def doTest(classIndex, pop,  dataset,classes,fold):
    Result=[]
    plt.figure() 
    for i in range(len(pop)):
        
      Result.append(calculateObjectiveValues(calculateValues(pop[i].getCandidateValuesRounded(),classIndex, dataset, classes)))
    file_exists(test_file)
    print("len(result):",len(Result))
    output_file = f"Experiments/Test/{fold}{classIndex}"  
    for i in range(len(Result)):
        plt.plot(Result[i][0], Result[i][1], 'bo')
        entry = [fold, classIndex, Result[i][0], Result[i][1]]
        append_to_csv(test_file, entry)

    plt.savefig(output_file)
    plt.close() 
  
    return Result

def PerformOperation(classIndex, pop, method, x_train,y_train, fold):
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
        pop.rungaOperations(x_train, y_train, threshold, classIndex)
        pop.checkDominance()
      
    
    
    return pop.printBestSolutions(classIndex, fold)

pop = PopulationNSGA(populationSize, MuCoefficient, mumCoefficient, crossoverProbability, mutationProbability)

########################################### For objectives 3 and 4 NSGA2

for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):  # For fold number
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
 
    
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    

    for classIndex in range(2):
        print("Fold Number:", fold, "Class:", classIndex, " Training")
        for i in range(populationSize):
            candidateID = 'a' + str(i)
            found = False
            while(not found):
                a = Candidate(candidateID, variableCount, lower, upper, X_train, y_train, threshold, classIndex)
               
               
                   
                if(a.getObjectiveValues()[0] > 0 or a.getObjectiveValues()[1] > 0):
                    pop.add_to_population(a)
                    found = True
                   
      
          
        # Training phase 
        
        BestTrain = PerformOperation(classIndex, pop, True, X_train,y_train, fold)
        for i in range(0, len(BestTrain)):
            print("Train Precision:", np.round(BestTrain[i].getObjectiveValues()[0], 3), "Recall:", np.round(BestTrain[i].getObjectiveValues()[1], 3))  
        
        print("Fold Number:", fold, "Class:", classIndex, " Test")
        
        BestTest = doTest(classIndex, BestTrain,  X_test,y_test,fold)
    
        for i in range(0, len(BestTest)):
            print("Test Precision:", np.round(BestTest[i][0], 3), "Recall:", np.round(BestTest[i][1], 3))    