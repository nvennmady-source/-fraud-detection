import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, RocCurveDisplay

# =====================
#  Genetic Algorithm (GA) Functions
# =====================
def init_population(n, c):
    return np.array([[math.ceil(e) for e in pop] for pop in (np.random.rand(n, c) - 0.5)]), np.zeros((2, c)) - 1

def single_point_crossover(population):
    r, c = population.shape[0], population.shape[1]
    n = np.random.randint(1, c)
    for i in range(0, r, 2):
        if i + 1 < r:
            population[i], population[i + 1] = (
                np.append(population[i][:n], population[i + 1][n:]),
                np.append(population[i + 1][:n], population[i][n:]),
            )
    return population

def flip_mutation(population):
    return population.max() - population

def random_selection(population):
    r = population.shape[0]
    new_population = population.copy()
    for i in range(r):
        new_population[i] = population[np.random.randint(0, r)]
    return new_population

def predictive_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    rf = RandomForestClassifier(n_estimators=200, random_state=10)
    rf.fit(X_train, y_train)
    return accuracy_score(y_test, rf.predict(X_test))

def compute_fitness(data, feature_list, target_var, population):
    fitness = []
    for i in range(population.shape[0]):
        columns = [feature_list[j] for j in range(population.shape[1]) if population[i, j] == 1]
        fitness.append(predictive_model(data[columns], data[target_var]))
    return np.array(fitness)

def memorize(pop, memory):
    return np.append(memory, pop.reshape(1, memory.shape[1]), axis=0)

def replace_duplicate(population, memory):
    for i in range(population.shape[0]):
        counter = 0
        while np.any(np.all(memory == population[i], axis=1)) and counter < 100:
            population[i] = np.array([math.ceil(k) for k in (np.random.rand(population.shape[1]) - 0.5)])
            counter += 1
        memory = memorize(population[i], memory)
    return population, memory

def ga(data, feature_list, target, n, max_iter):
    c = len(feature_list)
    population, memory = init_population(n, c)
    population, memory = replace_duplicate(population, memory)

    fitness = compute_fitness(data, feature_list, target, population)
    optimal_value = max(fitness)
    optimal_solution = population[np.where(np.atleast_1d(fitness) == optimal_value)][0]

    for _ in range(max_iter):
        population = random_selection(population)
        population = single_point_crossover(population)
        if np.random.rand() < 0.3:
            population = flip_mutation(population)

        population, memory = replace_duplicate(population, memory)
        fitness = compute_fitness(data, feature_list, target, population)

        if max(fitness) > optimal_value:
            optimal_value = max(fitness)
            optimal_solution = population[np.where(np.atleast_1d(fitness) == optimal_value)][0]

    return optimal_solution, optimal_value

# =====================
#  Main Pipeline
# =====================
if __name__ == "__main__":
    # Load dataset (update path accordingly)
    creditcard_df = pd.read_csv("creditcard.csv")

    X = creditcard_df.drop(["Class"], axis=1)
    y = creditcard_df["Class"]
    feature_list = list(X.columns)

    # Run GA for feature selection
    feature_set, acc_score = ga(creditcard_df.sample(n=20000), feature_list, "Class", n=16, max_iter=20)
    optimal_features = [feature_list[i] for i in range(len(feature_list)) if feature_set[i] == 1]

    print("Optimal Feature Set:\n", optimal_features)
    print("Optimal Accuracy =", round(acc_score * 100), "%")

    # Example predefined feature sets (from GA runs)
    F = [
        ['V1','V5','V7','V8','V11','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','Amount'],
        ['V1','V6','V13','V16','V17','V22','V23','V28','Amount'],
        ['V2','V11','V12','V13','V15','V16','V17','V18','V20','V21','V24','V26','Amount'],
        ['V2','V7','V10','V13','V15','V17','V19','V28','Amount'],
        ['Time','V1','V7','V8','V9','V11','V12','V14','V15','V22','V27','V28','Amount']
    ]

    X_train_in, X_test_in, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Loop through feature sets
    for v in range(len(F)):
        print(f"\nVector {v+1} ({len(F[v])} features) RESULTS =======")
        X_train, X_test = X_train_in[F[v]], X_test_in[F[v]]

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=150),
            "Decision Tree": DecisionTreeClassifier(),
            "ANN": MLPClassifier(hidden_layer_sizes=150, random_state=10, max_iter=500),
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(random_state=19, max_iter=500)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            print("==========================================")
            print(f"{name} Results:")
            print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2))
            print("Recall:", round(recall_score(y_test, y_pred) * 100, 2))
            print("Precision:", round(precision_score(y_test, y_pred) * 100, 2))
            print("F1-Score:", round(f1_score(y_test, y_pred) * 100, 2))

        # Plot ROC curves
        ax = plt.gca()
        for model in models.values():
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        fig_name = f"results/roc_auc_vector_{v+1}.png"
        plt.savefig(fig_name, dpi=300)
        plt.show()
