import numpy as np
import pandas as pd
from math import sqrt
import random

class SpamClassifier:
    def __init__(self, bias, lRate, epochs):
        #Initialise parameters for training the model
        
        self.random = np.random.default_rng()
        self.epochLoss = []
        self.weights = None
        self.bias = bias
        self.lRate = lRate
        self.epochs = epochs
        
    def train(self, data):
        #Training loop, works through each input at a time rather than in batches, calculates predictions and
        # adjusts weights and bias to improve accuracy
        #Loss is also calculated and the stored values can be displayed to show the loss curve through training
        
        for e in range(self.epochs):
            indvLoss = []
            predictions = []
            for i in range(len(data)):
                input = data[i][1:]
                response = data[i][0]
                wSum = self.weightedSum(input)
                prediction = self.sigmoid(wSum)
                loss = self.crossEntropy(response, prediction)
                indvLoss.append(loss)
                self.updateWeights(response, prediction, input)
                self.updateBias(response, prediction)
                if prediction < 0.5: #Prediction is a floating point and we want binary so it is changed to the closest one
                    correctedPred = 0
                else:
                    correctedPred = 1
                predictions.append((correctedPred, response))
            averageLoss = sum(indvLoss)/len(indvLoss)
            self.epochLoss.append(averageLoss)
            #print(averageLoss)
            #print("Prediction\inputNum")
            #print(predictions)

    def initialWeights(self, training_spam):
        #Xavier weight initialisation used to create random weights with a uniform probability distribution
        
        inputNum = len(training_spam[0])-1
        lower = -(1.0 / sqrt(inputNum))
        upper = (1.0 / sqrt(inputNum))
        weights = self.random.random(len(training_spam[0])-1)
        scaledWeights = lower + weights * (upper - lower)
        self.weights = scaledWeights

    def weightedSum(self, input):
        #Sum of all inputs multiplied by weights with the addition of bias
        
        return np.dot(input, self.weights) + self.bias

    def sigmoid(self, wSum):
        #Sigmoid function is the activation function, gives a more analog output in comparison to the step function
        
        return 1/(1+np.exp(-wSum))
    
    def sigmoidDeriv(self, X):
        #Sigmoid derivative is used to give the gradient of the slope, used in updating the weights
        
        return X * (1 - X)

    def crossEntropy(self, response, prediction):
        #Used only to calculate the loss for to be displayed for optimising of hyper-parameters
        
        return -(response*np.log10(prediction)+(1-response)*np.log10(1-prediction))

    def updateWeights(self, response, prediction, input):
        #Updating the weights by correcting them with the error calculated, sigmoid derivation and learning rate
        
        newWeights = []
        for x,w in zip(input, self.weights):
            newWeight = w + self.lRate*(response-prediction)*x*self.sigmoidDeriv(prediction)
            newWeights.append(newWeight)
        self.weights = newWeights
        
    def updateBias(self, response, prediction):
        #Updating the bias similarly to weights but only using the learning rate and the error
        
        self.bias = self.bias + self.lRate*(response-prediction)

    def predict(self, data):
        #Predicting the outputs for a new set of data using the weights and bias calculated from training
        
        predictions = []
        for i in range(len(data)):
            input = data[i]
            wSum = self.weightedSum(input)
            prediction = self.sigmoid(wSum)
            if prediction < 0.5:
                correctedPred = 0
            else:
                correctedPred = 1
            predictions.append(correctedPred)
        return np.asarray(predictions)

def create_classifier():
    #Setting the hyper-parameters and initialising an instance of SpamClassifier
    
    bias = 0
    lRate = 0.05
    epochs = 30
    
    classifier = SpamClassifier(bias, lRate, epochs)
    
    training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
    classifier.initialWeights(training_spam)
    
    """data = classifier.genData(training_spam)"""
    classifier.train(training_spam)
    return classifier

classifier = create_classifier()

SKIP_TESTS = False

if not SKIP_TESTS:
    testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
    test_data = testing_spam[:, 1:]
    test_labels = testing_spam[:, 0]

    predictions = classifier.predict(test_data)
    accuracy = np.count_nonzero(predictions == test_labels)/test_labels.shape[0]
    print(f"Accuracy on test data is: {accuracy}")