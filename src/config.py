import os

#Run the .py files from the main directory not from src directory
#Input paths
TRAINING_DATA = os.path.join(os.getcwd(), "input\\train.csv")
TRAINING_FOLD_DATA = os.path.join(os.getcwd(), "input\\train_folds.csv")
TEST_DATA = os.path.join(os.getcwd(), "input\\test.csv")

#Output paths
MODEL_OUTPUT = os.path.join(os.getcwd(), "models")

#Input details
TARGET_COL = ["Response"]
PROBLEM_TYPE = "binary_classification"
