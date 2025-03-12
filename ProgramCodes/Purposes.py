# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:19:33 2025

@author: cebrail
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 01:17:32 2021

@author: AsusSY
"""

import numpy as np

# This file contains objectives and necessary benchmark functions

# PROBLEM-SPECIFIC OBJECTIVES. THIS SECTION WILL CONTAIN OBJECTIVES RELATED TO SPECIFIC PROBLEMS

#####################################################################################
# Objectives 1 and 2 are a two-objective problem. It has 2 variables
# The first variable is between 0 and 1, the second variable is between 0 and 3
def precision(inputVariableValues):
    # precision value
    # ☻tp tn fp fn
    if(inputVariableValues[0] == 0):
        return 0
    else:
        intermediate = (inputVariableValues[0] + inputVariableValues[2])
        if intermediate > 0:
            return inputVariableValues[0] / intermediate
        else:
            return -999

def recall(inputVariableValues):
    if(inputVariableValues[0]==0):
       #☺ print(" rec:",gelenDegiskenDegerleri)
        return 0
    else:
#    return np.random.uniform(-5.0,5.1)
     # print("gelen degişken değerleri:"+str(gelenDegiskenDegerleri))
      intermediate=(inputVariableValues[0]+inputVariableValues[3])
      if intermediate>0:
          
         # print("rec:"+str(gelenDegiskenDegerleri[0]/ara))
          return inputVariableValues[0]/intermediate
      else:
          return -999
# Objectives 3 and 4 are a two-objective problem. It has 3 variables and each is between -5 and 5
def acc(inputVariableValues):
    # ☻tp tn fp fn
    intermediate = (inputVariableValues[0] + inputVariableValues[1])
    total = sum(inputVariableValues)
    if(total > 0):
        return intermediate / total

# Objectives 3 and 4 are a two-objective problem. It has 3 variables and each is between -5 and 5
def objective4(inputVariableValues):
    variableCount = len(inputVariableValues)
    X = inputVariableValues
    
    total = 0
    
    for i in range(variableCount - 1):
        total = total + np.abs(X[i]) ** (0.8) + 5 * (np.sin(X[i])) ** 3
        
    return total
#####################################################################################

################################ BENCHMARKS ############################

def DTLZ_1(candidate):
    pass