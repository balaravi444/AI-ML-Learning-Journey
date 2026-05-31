# Program 3 - MLModel Inheritance
# Day 11 - OOP: Inheritance & Method Overriding

# Parent class
class MLModel:
    def __init__(self, name, version, company):
        self.name = name
        self.version = version
        self.company = company

    def describe(self):
        print(f"Model: {self.name} (v{self.version}) created by {self.company}")


# Child class 1
class RandomForest(MLModel):
    def __init__(self, name, version, company, num_trees):
        super().__init__(name, version, company)
        self.num_trees = num_trees

    def predict(self):
        print(f"[{self.name}] voting across {self.num_trees} decision trees to make a prediction...")


# Child class 2
class NeuralNetwork(MLModel):
    def __init__(self, name, version, company, num_layers):
        super().__init__(name, version, company)
        self.num_layers = num_layers

    def predict(self):
        print(f"[{self.name}] passing data through {self.num_layers} hidden layers using complex math...")


# Creating objects
rf_model = RandomForest("ForestPredictor", 1.2, "Scikit-Learn", 100)
nn_model = NeuralNetwork("GPT-Vision", 4.0, "OpenAI", 120)

# Using methods
print("--- Random Forest Model ---")
rf_model.describe()
rf_model.predict()

print("\n--- Neural Network Model ---")
nn_model.describe()
nn_model.predict()
