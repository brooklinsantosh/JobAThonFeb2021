import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib 
import argparse

import config
import model_dispatcher


def run(fold, model):
    df = pd.read_csv(config.TRAINING_DATA)
    test_df = pd.read_csv(config.TEST_DATA)
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold==fold].reset_index(drop=True)

    ytrain = train_df.Response.values
    yvalid = valid_df.Response.values

    train_df = train_df.drop(["ID", "Response", "kfold"], axis=1)
    valid_df = valid_df.drop(["ID", "Response", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + test_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl
    
    clf = model_dispatcher.MODELS[model]
    clf.fit(train_df,ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    
    print(metrics.roc_auc_score(yvalid,preds))

    # joblib.dump(label_encoders, f"models/{MODEL}_{fold}_label_encoder.pkl")
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"{model}_{fold}.bin"))
    # joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)

    args = parser.parse_args()
    run(fold = args.fold,
        model = args.model)


