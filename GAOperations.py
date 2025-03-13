# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:18:10 2025

@author: cebrail
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 22:45:05 2021

@author: AsusSY
"""
import numpy as np
import string
import random
from Candidate import Candidate

class GAOperations:
    
    def __init__(self, muCoefficient, mumCoefficient, crossoverProbability, mutationProbability):
        
        self.muCoefficient = muCoefficient
        self.mumCoefficient = mumCoefficient
        self.crossoverProbability = crossoverProbability
        self.mutationProbability = mutationProbability
        
        self.NewCandidates = []
        
    # This is the main function that returns candidates generated as a result of crossover and mutation 
    def GenerateNewCandidates(self, parentPool, TrainingData, Classes, threshold, classIndex):
        
        self.parentPool = parentPool
        
        self.upperBounds = parentPool[0].getUpperBounds()
        self.lowerBounds = parentPool[0].getLowerBounds()
        
        self.NewCandidates.clear()
        
        for i in range(len(self.parentPool)):
            
            rr = np.random.uniform()
        
            # Perform crossover with 90% probability and mutation with 10% probability
            if(rr < self.crossoverProbability):
               
                candidate1, candidate2 = self.SBXGenerateCandidates(TrainingData, Classes, threshold, classIndex)
                self.NewCandidates.append(candidate1)
                self.NewCandidates.append(candidate2)
            
        else:
            candidate1 = self.CreateCandidateWithMutation(TrainingData, Classes, threshold, classIndex)
            self.NewCandidates.append(candidate1)

        return self.NewCandidates
        
    # Uses SBX for crossover. Each variable is calculated with a specific distribution
    def SBXGenerateCandidates(self, TrainingData, Classes, threshold, classIndex):
        # Use permutation technique for selection
        order = np.random.permutation(len(self.parentPool))
       
        parent1 = self.parentPool[order[0]]
        parent2 = self.parentPool[order[1]]
       
        # A random array is needed to create two children in SBX
        uSy = np.random.uniform(size=parent1.getNumVariables())
       
        # A second array will be used in the method
        bDz = np.zeros(parent1.getNumVariables())
       
        for i in range(parent1.getNumVariables()):
           
           if(uSy[i] <= 0.5):
               bDz[i] = (2 * uSy[i]) ** (1 / (self.muCoefficient + 1))
           else:
               bDz[i] = (1 / (2 * (1 - uSy[i]))) ** (1 / (self.muCoefficient + 1))
               
       # Calculate the variable values of the children
        c1D = 0.5 * (((1 + bDz) * parent1.getCandidateValues()) + ((1 - bDz) * parent2.getCandidateValues()))
        c2D = 0.5 * (((1 - bDz) * parent1.getCandidateValues()) + ((1 + bDz) * parent2.getCandidateValues()))
       
       # Perform boundary control on the generated values
        c1D = self.checkBounds(c1D)
        c2D = self.checkBounds(c2D)
       
       # Create new crossover candidates and set their new values
        cId = 'c' + self.generateRandomID()
        variableCount = parent1.getNumVariables()
        objectiveCount = parent1.getObjectiveCount()
        c1 = Candidate(cId, variableCount, self.lowerBounds, self.upperBounds, TrainingData, Classes, threshold, classIndex)
        c1.setCandidateValues(c1D)
  
        cId = 'c' + self.generateRandomID()
        variableCount = parent2.getNumVariables()
        objectiveCount = parent2.getObjectiveCount()
        c2 = Candidate(cId, variableCount, self.lowerBounds, self.upperBounds, TrainingData, Classes, threshold, classIndex)
        c2.setCandidateValues(c2D)
               
        return c1, c2
    
    # In mutation, each value of the selected candidate is changed and a new candidate is generated
    def CreateCandidateWithMutation(self, TrainingData, Classes, threshold, classIndex):
    
        # Select a random candidate
        a = np.random.randint(0, len(self.parentPool))
        candidate = self.parentPool[a]
        
        # Generate random array and delta array for mutation
        uSy = np.random.uniform(size=candidate.getNumVariables())
        delta = np.zeros(candidate.getNumVariables())
        
        # Calculate the delta array
        for i in range(candidate.getNumVariables()):
            
            if uSy[i] < 0.5:
                delta[i] = (2 * uSy[i]) ** (1 / (self.mumCoefficient + 1)) - 1
            else:
                delta[i] = 1 - (2 * (1 - uSy[i])) ** (1 / (self.mumCoefficient + 1))
                
        # Perform mutation by adding the delta value to the candidate's own values
        values = candidate.getCandidateValues()
        values = values + delta
        values = self.checkBounds(values)
        
        mId = 'm' + self.generateRandomID()
        variableCount = candidate.getNumVariables()
        objectiveCount = candidate.getObjectiveCount()
        m1 = Candidate(mId, variableCount, self.lowerBounds, self.upperBounds, TrainingData, Classes, threshold, classIndex)
        m1.setCandidateValues(values)
       
        return m1
    
    # Checks the boundary of the incoming variable values
    def checkBounds(self, dD):
        for i in range(len(dD)):
            if dD[i] > self.upperBounds[1]:
                dD[i] = self.upperBounds[1]
            
            if dD[i] < self.lowerBounds[0]:
                dD[i] = self.lowerBounds[0]
                
        return dD
        
    # Generates random IDs for candidates to avoid similarity
    def generateRandomID(self):
        letters = string.ascii_lowercase
        word = ''.join(random.choice(letters) for i in range(3))
        return word