import pandas as pd

from create_folds import CreateFolds
import config
from imputation import Imputation

df = pd.read_csv(config.TRAINING_DATA)
cf = CreateFolds(df, target_cols=config.TARGET_COL,
                    problem_type=config.PROBLEM_TYPE ,
                    shuffle = True)
df_split = cf.split()
print(df_split.kfold.value_counts())
print(df_split["Health Indicator"].isnull().sum())
imp = Imputation(df_split)
df_imp = imp.impute(impute_method="mode", cols=["Health Indicator"])
print(df_imp["Health Indicator"].isnull().sum())

