import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold
from PopulationNSGA import PopulationNSGA
from Candidate import Candidate

# Hiperparametrelerin test edilecek değerleri
population_sizes = [20, 50, 100]
crossover_rates = [0.7, 0.8, 0.9]
mutation_rates = [0.01, 0.05, 0.1]
max_iterations = [50, 100, 200]
# Veri kümesi yükleme
csv_file = "./DataSet/Syrian.csv"
df = pd.read_csv(csv_file, sep=';', lineterminator='\r').dropna()
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
variable_count = df.shape[1] - 1
threshold = round(((variable_count * 45) / 100), 2)
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Sonuçları saklamak için liste
results = []

# Hiperparametre kombinasyonlarını oluştur
param_combinations = list(itertools.product(population_sizes, crossover_rates, mutation_rates, max_iterations))

def PerformOperation(classIndex,max_iteration, pop, method, Dataset,Classes, fold, test):
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


def evaluate_hyperparams(pop_size, cx_rate, mut_rate, max_iter):
    
    pop = PopulationNSGA(pop_size,20,20, cx_rate, mut_rate)
    precision_list = []
    recall_list=[]
    for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = X_train.reset_index(drop=True)
        Classes = X_train.iloc[:, -1].reset_index(drop=True)
        
        for classIndex in range(2):
            for i in range(pop_size):
                candidateID = 'a' + str(i)
                found = False
                while not found:
                    a = Candidate(candidateID, variable_count, np.array([0, 0]), np.array([1, 1]), X_train, Classes, threshold, classIndex)
                    if a.getObjectiveValues()[0] > 0 or a.getObjectiveValues()[1] > 0:
                        pop.add_to_population(a)
                        found = True
                            
            BestTrain = PerformOperation(classIndex,max_iter, pop, True, X_train,Classes, fold, 0)
            BestTest = PerformOperation(classIndex,max_iter, pop, True, X_train,Classes, fold, 1)
            
            for i in range(len(BestTest)):
                precision_list.append(BestTest[i].getObjectiveValues()[0])
                recall_list.append(BestTest[i].getObjectiveValues()[1])

    return np.mean(precision_list), np.median(recall_list)

# Hiperparametreleri test et ve sonuçları kaydet
for params in param_combinations:
    pop_size, cx_rate, mut_rate, max_iter = params
    precision,recall = evaluate_hyperparams(pop_size, cx_rate, mut_rate, max_iter)
    results.append((params, precision,recall))
    print(f"Params: {params}, precision: {precision}, Recall: {recall}")

# En iyi hiperparametreleri belirle
best_params = max(results, key=lambda x: x[1])
print(f"Best Hyperparameters: {best_params[0]}, Best Performance: {best_params[1]}")

df_results = pd.DataFrame(results, columns=["Hyperparameters", "Precision", "Recall"])
df_results[["Population Size", "Crossover Rate", "Mutation Rate", "Max Iterations"]] = pd.DataFrame(df_results["Hyperparameters"].tolist(), index=df_results.index)
df_results.drop(columns=["Hyperparameters"], inplace=True)
print(df_results)

# Görselleştirme (crossover oranını vurgulayan bir çizgi grafiği)
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
for cx_rate in crossover_rates:
    subset = df_results[df_results["Crossover Rate"] == cx_rate]
    plt.plot(subset.index, subset["Precision"], marker='o', label=f"Crossover Rate: {cx_rate} (Precision)")
    plt.plot(subset.index, subset["Recall"], marker='s', linestyle='dashed', label=f"Crossover Rate: {cx_rate} (Recall)")

plt.xlabel("Hyperparameter Combination Index")
plt.ylabel("Performance")
plt.title("NSGA-II Hyperparameter Analysis (Crossover Rate Effect on Precision & Recall)")
plt.legend()
plt.show()
