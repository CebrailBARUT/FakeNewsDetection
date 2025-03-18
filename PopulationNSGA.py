# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:11:25 2025

@author: cebrail
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 21:41:05 2021

@author: AsusSY
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from GAOperations import GAOperations
import Purposes as benchmarks
import random
import csv
from os import path
import copy as cp


class PopulationNSGA:

    def __init__(self, capacity, mu_coefficient, mum_coefficient, crossover_probability, mutation_probability):
        self.capacity = capacity
        self.population = []
        self.parent_pool = []
        self.last_front = 0
        self.last_front_list = []
        self.allFrontLists = []  # contains lists
        self.objective_values_min = []
        self.objective_values_max = []
        self.objective_count = 2
        self.objective_functions = ['precision', 'recall']
        self.mu_coefficient = mu_coefficient
        self.mum_coefficient = mum_coefficient
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.bestsolution=[]
    def getobjectiveCount(self):
        return self.objective_count
    def element_exists(self,lst, element):
      
     
      for i in lst:
          if i.getID()==element.getID():
             
              return True
          
     
      return False
  
    def setBestSolution(self,solutions):
        self.bestsolution=solutions
    def getBestSolution(self):
        return self.bestsolution    
    def file_exists(self, file):
        if not path.exists(file):
            with open(file, mode='a', newline='') as file_writer:
                header = ["Fold", "Class", "Precision", "Recall"]
                writer = csv.writer(file_writer)
                writer.writerow(header)
    

    def append_to_csv(self, file, element):
        with open(file, 'a+', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerow(element)

    def add_to_population(self, new_candidate):
        self.population.append(new_candidate)

    def add_new_list_to_population(self, candidate_list):
        self.population.extend(candidate_list)

    def removefromPupdateopulation(self, candidate_id):
        self.population = [candidate for candidate in self.population if candidate.getID() != candidate_id]

    def updatePopulation(self, new_population):
        self.population.clear()
        self.population.extend(new_population)
        self.last_front = -1
        self.last_front_list.clear()
        self.allFrontLists.clear()
        self.parent_pool.clear()
    

# In the objectives and benchmarks file
# A string list containing the names of which objective functions will be used will come here
# The desired function can be called using getattr()
    def findObjectiveValues(self, objective_list):
        self.objective_count = len(objective_list)

        for candidate in self.population:
            values = np.zeros(self.objective_count)

            for i in range(len(objective_list)):
                func = getattr(benchmarks, objective_list[i])
                values[i] = func(candidate.get_tptnfpfn())

            candidate.set_objective_function_values(values)

    def get_population(self):
        return self.population

    def checkDominance(self, is_maximization=True):
        for candidate in self.population:
            candidate.clearDominanceLists()

        for candidate1 in self.population:
            for candidate2 in self.population:
                if candidate1.getID() != candidate2.getID():
                    if is_maximization:
                        result = candidate1.isDominatedbyGreater(candidate2)
                    else:
                        result = candidate1.isDominatedbySmaller(candidate2)

                    if result:
                        candidate1.addDominatingCandidate(candidate2.getID())

            candidate1.setFrontNumber(len(candidate1.getDominatingCandidates()) + 1)

    def needs_special_selection(self):
        self.sort_population_by_front()
        if self.capacity >= self.get_current_population_size():
            self.last_front = -1
            return False

        last_candidate = self.population[self.capacity - 1]
        next_candidate = self.population[self.capacity]

        if last_candidate.getFrontNumber() == next_candidate.getFrontNumber():
            self.last_front = last_candidate.getFrontNumber()
            return True

        self.last_front = -1
        return False

    def split_fronts(self):
        self.SortPopulationByFront()
        self.allFrontLists.clear()
        front_number = self.population[0].getFrontNumber()
        candidates_list = []

        for i, candidate in enumerate(self.population):
            if candidate.getFrontNumber() == front_number:
                candidates_list.append(candidate.getCandidateCopy())
                if (i + 1) == len(self.population):  # last element is single
                    self.allFrontLists.append(copy.deepcopy(candidates_list))
            else:
                self.allFrontLists.append(copy.deepcopy(candidates_list))
                candidates_list.clear()
                front_number = candidate.getFrontNumber()
                candidates_list.append(candidate.getCandidateCopy())

                if (i + 1) == len(self.population):  # last element is single
                    self.allFrontLists.append(copy.deepcopy(candidates_list))

    def getCandidatesinLastFront(self):
        if self.last_front != -1:
            self.last_front_list.clear()

            for candidate in self.population:
                if candidate.getFrontNumber() == self.last_front:
                    self.last_front_list.append(candidate.getCandidateCopy())

        return self.last_front_list

    def show_last_front_list(self):
        # self.getCandidatesinLastFront()()
        print(self.getCandidatesinLastFront())

    def getCandidatesFromLastFront(self, count):
        candidates_list = []

        if self.last_front != -1:
            self.getCandidatesinLastFront()
            self.sort_byCrowdingDistanceLevel(self.last_front_list)

            for i in range(count):
                copy_candidate = self.last_front_list[i].getCandidateCopy()
                if not self.isinNewPopulation(copy_candidate, candidates_list):
                    candidates_list.append(copy_candidate)
                else:
                    i -= 1

        return candidates_list

    def findMaxMinInObjectives(self, front_list):
       
        objective_count = front_list[0].getobjectiveCount()
     

        self.objective_values_max.clear()
        self.objective_values_min.clear()

        for f in range(objective_count):
            max_value = -np.inf
            min_value = np.inf

            for sf in front_list:
                if max_value < sf. getObjectiveValues()[f]:
                    max_value = sf. getObjectiveValues()[f]

                if min_value > sf. getObjectiveValues()[f]:
                    min_value = sf. getObjectiveValues()[f]

            self.objective_values_max.append(max_value)
            self.objective_values_min.append(min_value)

        return self.objective_values_max, self.objective_values_min

    def calculate_crowding_distance(self):
        objective_count = self.population[0].getobjectiveCount()
        self.population.clear()

        for j in range(len(self.allFrontLists)):
            front_list = self.allFrontLists[j]
            max_values, min_values = self.findMaxMinInObjectives(front_list)

            for obj in range(objective_count):
                front_list = self.sortbyObjective(obj, front_list)
                for i in range(len(front_list)):
                    front_list[i].setCrowdingDistance(0)
                    if front_list[i].getObjectiveValues()[obj] == max_values[obj]:
                        front_list[i].setCrowdingDistance(np.inf)
                    elif front_list[i].getObjectiveValues()[obj] == min_values[obj]:
                        front_list[i].setCrowdingDistance(np.inf)
                    else:
                        previous = front_list[i - 1].getObjectiveValues()[obj]
                        next_value = front_list[i + 1].getObjectiveValues()[obj]
                        distance = front_list[i].getCrowdingDistance() + np.abs(next_value - previous) / np.abs(max_values[obj] - min_values[obj])
                        front_list[i].setCrowdingDistance(distance)

            self.population.extend(front_list)
        self.allFrontLists.clear()
    def CalculateCrowdingDistanceLevel(self):

            # Amaç sayısını koru
            numofObject = 2
    
  
 
            for k in range(len(self.allFrontLists)):
                
               
                level=1
              
                fListe = self.allFrontLists[k]
                
               
                n=len(fListe)
                if n <= 2:
                    for individual in fListe:
                        individual.setCrowdingDistance(float('inf'))  # Uç noktalar sonsuz
                        return

               
                for i in range(n):
                    fListe[i].setCrowdingDistance(0) 
                  
                for amc in range(numofObject):  # Her amaç için dön
                         obj_values = [ind.getObjectiveValues()[amc] for ind in fListe]  
                         max_val = max(obj_values)  
                         min_val = min(obj_values)  

       
                extremes = [ind for ind in fListe if ind.getObjectiveValues()[amc] in (max_val, min_val)]
                remaining = [
    ind for ind in fListe 
    if not any(all(ind.getObjectiveValues()[i] == ext.getObjectiveValues()[i] for i in range(len(ind.getObjectiveValues()))) for ext in extremes)
]


                for t in range(len(extremes)):
                    extremes[t].setCrowdingDistanceLevel(0)
                  
               
                maximum=0
                for m in range(len(remaining)):
                    
                    for s in range(len(remaining)):
                        # Candidate has no Crowding distance Level
                        if(remaining[s].getCrowdingDistanceLevel()==-1):
                             for j in range (len(extremes)):
                                        
                                for amc in range(numofObject):
                                         remaining[s].setCrowdingDistance(remaining[s].getCrowdingDistance()+ np.abs(remaining[s].getObjectiveValues()[amc]-extremes[j].getObjectiveValues()[amc]))
                               
                             maximum=-1
                             indis=-1
                            #Find maximum Crowding Distance in remaiining Candidate
                             for p in range(len(remaining)):
                               
                                if(remaining[p].getCrowdingDistance()>maximum):
                                               maximum=remaining[p].getCrowdingDistance()
                                               indis=p
                            
                                               #Assign Crowding Distance Level to Candidate has maximum Crowding distance indis=p
                             if(indis>-1):  
                                extremes.append(remaining[indis])
                                remaining[indis].setCrowdingDistanceLevel(level)
                              
                                level=level+1
                                maximum=-1
                                indis=-1
                                for g in range(len(remaining)):
                                    remaining[g].setCrowdingDistance(0)
               
                     
                        
                           
                        
                                           

            
        

   
    def IsSpecialSelectionNeeded(self):
        self.SortPopulationByFront()
    
        if(self.capacity >= self.getCurrentPopulationSize()):
            self.lastFront = -1
            return False

        lastCandidate = self.population[self.capacity - 1]
        nextCandidate = self.population[self.capacity]
    
        if(lastCandidate.getFrontNumber() == nextCandidate.getFrontNumber()):
            self.lastFront = lastCandidate.getFrontNumber()
            return True
    
        self.lastFront = -1
        return False
# Create a parent pool for crossover. It is always of even size
    def createParentPool(self):
        self.parent_pool.clear()
        self.SortPopulationByFront()

        pool_size = int(self.capacity / 2)

        for i in range(pool_size):
            random_indices = np.random.permutation(len(self.population))
            candidate1 = self.population[random_indices[0]]
            candidate2 = self.population[random_indices[1]]

            if candidate1.getFrontNumber() < candidate2.getFrontNumber():
                self.parent_pool.append(copy.deepcopy(candidate1))
            elif candidate1.getFrontNumber() > candidate2.getFrontNumber():
                self.parent_pool.append(copy.deepcopy(candidate2))
            else:
                if candidate1.getCrowdingDistanceLevel() < candidate2.getCrowdingDistanceLevel():
                    self.parent_pool.append(copy.deepcopy(candidate1))
                else:
                    self.parent_pool.append(copy.deepcopy(candidate2))

    def rungaOperations(self, training_data, classes, threshold, class_label):
        ga_operations = GAOperations(self.mu_coefficient, self.mum_coefficient, self.crossover_probability, self.mutation_probability)
        candidates = ga_operations.GenerateNewCandidates(self.parent_pool, training_data, classes, threshold, class_label)
        self.population.extend(candidates)

    def printParentPool(self):
        np.set_printoptions(precision=4)
        print('----- PARENT Values ------')
        for candidate in self.parent_pool:
            print(candidate.getID(), 'F=', candidate.getFrontNumber(), 'CD=', candidate.getCrowdingDistanceLevel(), ' for D=', candidate.get_candidate_values(),
                  ' A=', candidate.getObjectiveValues())
        print('----------------')
        print('')

    def SortPopulationByFront(self):
        for candidate in self.population:
            candidate.setSortingCriteria(-1)

        self.population = sorted(self.population)

    def sortbyObjective(self, objective_index, front_list):
        for candidate in front_list:
            candidate.setSortingCriteria(objective_index)

        return sorted(front_list)

    def sort_byCrowdingDistanceLevel(self, front_list):
        for candidate in front_list:
            candidate.setSortingCriteria(-2)

        return sorted(front_list, reverse=True)

    def createNewGeneration(self, special_selection_needed):
        new_population = []
        self.SortPopulationByFront()

        if special_selection_needed:
            first_candidate = self.population[0]
            last_candidate = self.population[self.capacity - 1]

            if first_candidate.getFrontNumber() == last_candidate.getFrontNumber():
                self.split_fronts()
                front_list = self.allFrontLists[0]
                front_list = self.sort_byCrowdingDistanceLevel(front_list)

                for i in range(self.capacity):
                    copy_candidate = front_list[i].getCandidateCopy()
                    if not self.isinNewPopulation(copy_candidate, new_population):
                        new_population.append(copy_candidate)

                return new_population

            for candidate in self.population:
                if candidate.getFrontNumber() != self.last_front:
                    new_population.append(candidate.getCandidateCopy())
                else:
                    break

            count = self.capacity - len(new_population)
            others = self.getCandidatesFromLastFront(count)
            new_population.extend(others)
        else:
            for i in range(self.capacity):
                new_population.append(self.population[i].getCandidateCopy())

        return new_population

    def isinNewPopulation(self, copy_candidate, new_population):
        for candidate in new_population:
            if (candidate.getCandidateValues() == copy_candidate.getCandidateValues()).all():
                return True
        return False

    def printPopulation(self):
        self.sort_population_by_front()
        print('----- Candidate Objective Values ------')
        np.set_printoptions(precision=4)
        for candidate in self.population:
            print(candidate.getID(), 'F=', candidate.getFrontNumber(), 'CD=', candidate.getCrowdingDistanceLevel(), ' for D=', candidate.getCandidateValues(),
                  ' A=', candidate.getObjectiveValues())
        print('----------------')
        print('')

 

    def printDominatingCandidates(self):
        print('----- Candidates Dominated By ------')
        for candidate in self.population:
            print(candidate.get_candidate_id(), '=', candidate.get_dominating_candidates(),
                  ' FrontNO=', candidate.get_front_number())
        print('----------------')
        print('')

    def printBestSolutions(self, class_label, fold):
        self.split_fronts()
        return self.draw_best_solutions(class_label, fold)

    def draw_best_solutions(self, class_label, fold):
        output_file = f"Experiments/Train/{fold}{class_label}"
        csv_file = "Experiments/Train/Train.csv"
        bst_sol=[]
        plt.clf()
        self.file_exists(csv_file)
        for candidate in self.allFrontLists[0][:self.capacity]:
            plt.plot(candidate.getObjectiveValues()[0], candidate.getObjectiveValues()[1], 'bo')
            bst_sol.append(candidate.getObjectiveValues())
            entry = [fold, class_label, candidate.getObjectiveValues()[0], candidate.getObjectiveValues()[1]]
            self.append_to_csv(csv_file, entry)

        plt.savefig(output_file)
        plt.close()
        self.setBestSolution(bst_sol)
        return self.allFrontLists[0][:self.capacity]
    def get_best_solution(self, class_label, fold):
        results=[]
        for candidate in self.allFrontLists[0][:self.capacity]:
            results.append(candidate.getObjectiveValues()[0], candidate.getObjectiveValues()[1])
           
        return results   
           
        
    def getBestSolutions(self):
        
        sFront = self.allFrontLists[0][:self.capacity]
        return sFront

    ###################### GETTERS ##############

    
        
    def getLastFront(self):
        return self.last_front

    def getCurrentPopulationSize(self):
        return len(self.population)

    def getTotalCapacity(self):
        return self.capacity

    def get_parent_pool(self):
        return self.parent_pool

    def getAllFrontsList(self):
        print(self.allFrontLists)