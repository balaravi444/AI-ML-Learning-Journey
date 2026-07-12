# Program 3 - Abstract Classes
# Day 12 - OOP: Encapsulation, Polymorphism & Abstract Classes

from abc import ABC, abstractmethod


class MLModel(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def describe(self):
        print("I am an ML Model")


class LinearRegression(MLModel):
    def train(self):
        print("Training with straight line math...")

    def predict(self):
        print("Predicting using linear equation...")


class RandomForest(MLModel):
    def train(self):
        print("Training 100 decision trees...")

    def predict(self):
        print("Voting across all trees...")


# Creating objects
lr_model = LinearRegression()
rf_model = RandomForest()

print("--- Linear Regression ---")
lr_model.describe()
lr_model.train()
lr_model.predict()

print("\n--- Random Forest ---")
rf_model.describe()
rf_model.train()
rf_model.predict()
