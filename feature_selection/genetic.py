"""
Genetic Algorithm for Machine Learning model.
"""

import random

import numpy as np
from sklearn.metrics import accuracy_score


class Genetic:
    """
    Genetic Algorithm implementation, with 2-point crossover and bit-inversion mutation

    Attributes
    ----------
    __populationSize : int
        size of the population to fit on
    __noOfFeatures : int
        no of features in the dataset
    __noOfParents : int
        no of parents to select for the crossover
    __mutationRate : float
        rate at which the mutation will occur
    __noOfGenerations : int
        no of generations to run
    __model : model
        model to use for fit and score
    __X_train : data
        training data
    __X_test ; data
        testing data
    __y_train : label
        training labels
    __y_test : label
        testing labels

    Examples
    ------
    >>> model = GaussianNB()
    >>> genetic = Genetic(model=model, nFeatures=30, nParents=100, pSize=100, mutationRate=0.10, nGeneration=45)
    >>> chromosome, score = genetic.getChromosomeScore(X_train, X_test, y_train, y_test)
    >>> model.fit(X_train.iloc[:, chromosome], y_train)
    >>> predictions = model.predict(X_test.iloc[:, chromosome])
    """

    def __init__(self,
                 nFeatures,
                 pSize=100,
                 nParents=100,
                 mutationRate=0.10,
                 nGeneration=100,
                 model=None):

        # init some data
        self.__populationSize = pSize
        self.__noOfFeatures = nFeatures
        self.__noOfParents = nParents
        self.__mutationRate = mutationRate
        self.__noOfGenerations = nGeneration
        self.__model = model

        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None

    def __initializePopulation(self):
        """Selecting random features"""
        population = list()
        for i in range(self.__populationSize):
            chromosome = np.ones(self.__noOfFeatures, dtype=np.bool)
            chromosome[:int(0.3 * self.__noOfFeatures)] = False
            np.random.shuffle(chromosome)
            population.append(chromosome)

        return population

    def __getFitnessForPopulation(self, population):
        """
        Return the scores and predictions on the population

        Parameters
        ----------
        population : data
            generated data using the population

        Returns
        -------
        lol
            scores run
        lol
            predictions

        """
        scores = list()
        for chromosome in population:
            if all(x is False for x in chromosome):
                continue

            self.__model.fit(self.__X_train.iloc[:, chromosome], self.__y_train)
            predictions = self.__model.predict(self.__X_test.iloc[:, chromosome])
            scores.append(accuracy_score(self.__y_test, predictions))

        scores, predictions = np.array(scores), np.array(population)
        indices = np.argsort(scores)

        return list(scores[indices][::-1]), list(predictions[indices, :][::-1])

    def __selection(self, population):
        """
        Select parents for cross over

        Parameters
        ----------
        population : data
            features

        Returns
        -------
        data
            selected top feature
        """
        return population[0: self.__noOfParents]

    def __crossOver(self, population):
        """
        Two point cross over on the data

        Parameters
        ----------
        population : data
            features

        Returns
        -------
        data
            crossed population
        """
        newPopulation = population

        for i in range(self.__noOfParents):
            chromosome = population[i]
            chromosome[3: 7] = population[(i + 1) % len(population)][3: 7]
            newPopulation.append(chromosome)

        return newPopulation

    def __mutation(self, population):
        """
        Bit inversion mutation

        Parameters
        ----------
        population : data
            crossed over population

        Returns
        -------
        data
            features after mutation
        """
        newPopulation = list()

        for i in range(0, len(population)):
            chromosome = population[i]
            for j in range(len(chromosome)):
                if random.random() < self.__mutationRate:
                    chromosome[j] = not chromosome[j]
            newPopulation.append(chromosome)

        return newPopulation

    def getChromosomeScore(self, X_train, X_test, y_train, y_test):
        """
        Run the feature selection for gen times

        Parameters
        ----------
        X_train : data
            training set
        X_test : data
            testing data
        y_train : labels
            training labels
        y_test : labels
            testing labels

        Returns
        -------
        data
            best features
        score
            best scores

        """
        self.__X_train = X_train
        self.__X_test = X_test
        self.__y_train = y_train
        self.__y_test = y_test

        bestChromosome = None
        bestScore = None

        population = self.__initializePopulation()
        for i in range(self.__noOfGenerations):
            score, population = self.__getFitnessForPopulation(population)
            print(f"Generation {i} :: {score[0]}")

            population = self.__selection(population)
            population = self.__crossOver(population)
            population = self.__mutation(population)

            bestChromosome = population[0]
            bestScore = score[0]

        return bestChromosome, bestScore
