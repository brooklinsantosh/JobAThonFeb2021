import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import config

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.getcwd(), "input\\train_Df64byy.csv"))
    df["kfold"] = -1

    df = df.sample(frac = 1).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    for f, (t_,v_) in enumerate(kf.split(X=df,y=df.Response.values)):
        df.loc[v_,'kfold'] = f

    df.to_csv(config.TRAINING_DATA,index=False)