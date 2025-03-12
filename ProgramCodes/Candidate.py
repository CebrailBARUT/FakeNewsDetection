# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 21:58:05 2021

@author: AsusSY
"""

import numpy as np
import copy
import Purposes as objectives


class Candidate:
    
    def __init__(self, candidate_id, num_variables, lower_bound_matrix, upper_bound_matrix, dataset, classes, threshold, class_label):
        self.candidate_id = candidate_id
        self.num_variables = num_variables
        self.lower_bound_matrix = lower_bound_matrix
        self.upper_bound_matrix = upper_bound_matrix
        self.threshold = threshold
        self.objective_functions = ['precision', 'recall']
        self.classes = classes
        self.objective_values = np.zeros(len(self.objective_functions))
        self.candidate_values=np.zeros(num_variables)
        self.candidate_rounded_values = np.zeros(num_variables)
        self.dataset = dataset
        self.dominated_candidates = []
        self.dominating_candidates = []
        self.front_number = 0
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        
        self.num_objectives = len(self.objective_functions)
        self.sort_by_objective_index = -1
        self.class_label = class_label
        
        self.crowding_distance = 0
        self.crowding_distance_level = -1
        self.createCandidate()
        self.calculateValues(dataset, classes)
        self.calculateObjectiveValues()
    def CandidateisSame(self,cnd):
        
        return self.candidate_values_rounded == cnd.getCandidateValuesRounded()
 
    def createCandidate(self):
        # values=np.zeros(self.num_variables)
        values = np.random.random(self.num_variables)
        self.candidate_values = values
        self.candidate_rounded_values = np.round(values)
        # print("rounded value:",self.candidate_values_rounded)
        
    def findSimilarity(self, other):
        A = self.getCandidateValuesRounded()
        return np.sum(A == other)
    
    def calculateValues(self, dataset, classes):
       
        for j in range(len(dataset)):
           
            current_data = np.array(dataset.iloc[j].values.tolist(), int)
            similarity_score = self.findSimilarity(current_data)
            
            if similarity_score > self.threshold and classes[j] == self.class_label:
                self.incrementTruePositive()
               
            elif similarity_score > self.threshold and classes[j] != self.class_label:
                self.incrementFalsePositive()
                
            elif similarity_score < self.threshold and classes[j] == self.class_label:
                self.incrementFalseNegative()
                
            else:
                self.incrementTrueNegative()
              
       
       
    def calculateObjectiveValues(self):
        values = np.zeros(self.num_objectives)
      
        for i in range(self.num_objectives):
            func = getattr(objectives, self.objective_functions[i])
            
            values[i] = func(self.getTpfpfn())
        
        
        self.setObjectiveValues(values)
     
       
        
    def addDominatingCandidate(self, dominating_candidate_id):
        self.dominating_candidates.append(dominating_candidate_id)
      
    def addDominatedCandidate(self, dominated_candidate_id):
        self.dominated_candidates.append(dominated_candidate_id)
        
    def dominatesGreater(self, other):
        return np.any(self.objective_values > other.getObjectiveValues()) and not np.any(other.getObjectiveValues() > self.objective_values)
       
    def dominatesSmaller(self, other):
        return np.any(self.objective_values < other.getObjectiveValues()) and not np.any(other.getObjectiveValues() < self.objective_values)
    
    def isDominatedbyGreater(self, other):
        return np.any(other.getObjectiveValues() > self.objective_values) and not np.any(self.objective_values > other.getObjectiveValues())
       
    def isDominatedbySmaller(self, other):
        return np.any(other.getObjectiveValues() < self.objective_values) and not np.any(self.objective_values < other.getObjectiveValues())
        
    def clearDominanceLists(self):
        self.dominating_candidates.clear()
        self.dominated_candidates.clear()
        
    def incrementTruePositive(self):
        self.true_positive += 1
        
    def incrementTrueNegative(self):
        self.true_negative += 1
        
    def incrementFalsePositive(self):
        self.false_positive += 1
        
    def incrementFalseNegative(self):
        self.false_negative += 1
        
    def calculateObjectiveValuesFromList(self, objective_list):
        self.num_objectives = len(objective_list)
        values = np.zeros(self.num_objectives)
        for i in range(len(objective_list)):
            func = getattr(objectives, objective_list[i])
            values[i] = func(self.get_tpfpfn())
        
        self.set_objective_values(values)
        return 1 if np.all(values != 0) else 0
        
    ######################################## GETTERS ########
    
    def getID(self):
        return self.candidate_id
    
    def getCandidateValues(self):
        return copy.deepcopy(self.candidate_values)
    def getCandidateValuesRounded(self):
        return copy.deepcopy(self.candidate_rounded_values)
    
    
    def getObjectiveValues(self):
        return copy.deepcopy(self.objective_values)
    def getObjectiveCount(self):
        return len(self.objective_values)
    
    
    def getLowerBounds(self):
        return self.lower_bound_matrix
    
    def getUpperBounds(self):
        return self.upper_bound_matrix
    
    def getFrontNumber(self):
        return copy.deepcopy(self.front_number)
    
    def getDominatedCandidates(self):
        return copy.deepcopy(self.dominated_candidates)
    
    def getDominatingCandidates(self):
        return copy.deepcopy(self.dominating_candidates)
    
    def getNumberofDominatedCandidates(self):
        return len(self.dominated_candidates)
    
    def getNumberofDominatingCandidates(self):
        return len(self.dominating_candidates)
    
    def getNumObjectives(self):
        return self.num_objectives
    
    def getNumVariables(self):
        return self.num_variables
    
    def getCrowdingDistance(self):
        return round(self.crowding_distance, 3)
    
    def getCrowdingDistanceLevel(self):
        return round(self.crowding_distance_level, 3)
    
    def getPrecision(self):
        return self.precision
    
   
    
    def getRecall(self):
        return self.recall
   
    def getSortingCriteria(self):
        return self.sort_by_objective_index
    
    def getTruePositive(self):
        return self.true_positive
    
    def getTrueNegative(self):
        return self.true_negative
    
    def getFalsePositive(self):
        return self.false_positive
    
    def getFalseNegative(self):
        return self.false_negative
    
    def getTpfpfn(self):
        return [self.getTruePositive(), self.getTrueNegative(), self.getFalsePositive(), self.getFalseNegative()]
    
    def getTotal_tn_tp(self):
        return self.getTruePositive() + self.getTrueNegative()
    
    def getTotal_tp_fn(self):
        return self.getTruePositive() + self.getFalseNegative()
    
    def getTotal_tp_fp(self):
        return self.getTruePositive() + self.getFalsePositive()
    
    ######################################### SETTERS ################# 
    def setCandidateValues(self, new_values):
        self.candidate_values = new_values
        self.candidate_values_rounded = np.round(new_values, 0)
       
    def setObjectiveValues(self, new_objective_values):
        self.objective_values = new_objective_values
        
    def getobjectiveCount(self):
        return len(self.objective_values)
   
    
    def setFrontNumber(self, new_front_number):
        self.front_number = new_front_number
        
    def setDominatedCandidates(self, new_values):
        self.dominated_candidates = new_values
    
    def setDominatingCandidates(self, new_values):
        self.dominating_candidates = new_values
        
    def setCcrowdingDistanceLevel(self, value):
        self.crowding_distance_level = value   
    
    def setCrowdingDistance(self, cd):
        self.crowding_distance = cd
    def setCrowdingDistanceLevel(self, cdl):
        self.crowding_distance_level = cdl    
    def setPrecision(self, value):
        self.precision = value
        
   
        
    def setRecall(self, value):
        self.recall = value      
    
    def setSortingCriteria(self, criteria):
        self.sort_by_objective_index = criteria

    def setTruePositive(self, value):
        self.true_positive = value
        
    def setTrueNegative(self, value):
        self.true_negative = value
        
    def setFalsePositive(self, value):
        self.false_positive = value
        
    def setFalseNegative(self, value):
        self.false_negative = value
        
    def reset_values(self):
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        
    ######################### Comparison Functions for Sorting ##########
    
    def __eq__(self, other):
        if self.sort_by_objective_index == -1:
            return self.getFrontNumber() == other.getFrontNumber()
        elif self.sort_by_objective_index == -2:
            return self.getCrowdingDistanceLevel() == other.getCrowdingDistanceLevel()
        else:
            return self.getObjectiveValues()[self.sort_by_objective_index] == other.getObjectiveValues()[self.sort_by_objective_index]
    
    def __lt__(self, other):
        if self.sort_by_objective_index == -1:
            return self.getFrontNumber() < other.getFrontNumber()
        elif self.sort_by_objective_index == -2:
            return self.getCrowdingDistanceLevel() > other.getCrowdingDistanceLevel()
        else:
            return self.getObjectiveValues()[self.sort_by_objective_index] < other.getObjectiveValues()[self.sort_by_objective_index]
    
    def __le__(self, other):
        if self.sort_by_objective_index == -1:
            return self.getFrontNumber() <= other.getFrontNumber()
        elif self.sort_by_objective_index == -2:
            return self.getCrowdingDistanceLevel() >= other.getCrowdingDistanceLevel()
        else:
            return self.get_objective_values()[self.sort_by_objective_index] <= other.getObjectiveValues()[self.sort_by_objective_index] 
    
    def __gt__(self, other):
        if self.sort_by_objective_index == -1:
            return self.getFrontNumber() > other.getFrontNumber()
        elif self.sort_by_objective_index == -2:
            return self.getCrowdingDistanceLevel() < other.getCrowdingDistanceLevel()
        else:
            return self.get_objective_values()[self.sort_by_objective_index] > other.getObjectiveValues()[self.sort_by_objective_index] 
    
    def __ge__(self, other):
        if self.sort_by_objective_index == -1:
            return self.getFrontNumber() >= other.getFrontNumber()
        elif self.sort_by_objective_index == -2:
            return self.getCrowdingDistanceLevel() <= other.getCrowdingDistanceLevel()
        else:
            return self.getObjectiveValues()[self.sort_by_objective_index] >= other.getObjectiveValues()[self.sort_by_objective_index] 
    
    def __ne__(self, other):
        if self.sort_by_objective_index == -1:
            return self.getFrontNumber() != other.getFrontNumber()
        elif self.sort_by_objective_index == -2:
            return self.getCrowdingDistanceLevel() != other.getCrowdingDistanceLevel()
        else:
            return self.get_objective_values()[self.sort_by_objective_index] != other.getObjectiveValues()[self.sort_by_objective_index] 
    
    def __repr__(self):
        return self.getID()
        
    def getCandidateCopy(self):
        candidate_copy = Candidate(self.getID(), self.getNumVariables(), self.getLowerBounds(), self.getUpperBounds(), self.dataset, self.classes, self.threshold, self.class_label)
        
        candidate_copy.setCandidateValues(self.getCandidateValues())
        candidate_copy.setObjectiveValues(self.getObjectiveValues())
        candidate_copy.setFrontNumber(self.getFrontNumber())
        candidate_copy.setDominatedCandidates(self.getDominatedCandidates())
        candidate_copy.setDominatingCandidates(self.getDominatingCandidates())
        candidate_copy.setCrowdingDistance(self.getCrowdingDistance())
        candidate_copy.setCcrowdingDistanceLevel(self.getCrowdingDistanceLevel())
        
        return candidate_copy