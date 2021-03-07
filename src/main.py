import pandas as pd

from create_folds import CreateFolds
import config

df = pd.read_csv(config.TRAINING_DATA)
cf = CreateFolds(df, target_cols=config.TARGET_COL,
                    problem_type=config.PROBLEM_TYPE ,
                    shuffle = True)
df_split = cf.split()
print(df_split.kfold.value_counts())

