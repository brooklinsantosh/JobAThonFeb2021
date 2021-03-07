import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

import config

class CreateFolds:
    """
    Class to Create folds for different types of problems.
    Split method will split the dataset babsed on the inputs given.
    It will return the dataset with a column called kfold.
    There is a option to save the dataset as well

    problem_type can be binary_classification, multiclass_classification, multilabel_classification,
    single_col_regression, multiclass_classification, holdout_20
    target_cols is a list
    """
    def __init__(
            self, 
            df, 
            target_cols, 
            shuffle,
            problem_type="binary_classification", 
            multilabel_delimiter = ",",
            num_folds =5,
            random_state = 42,
            save = True,):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.multilabel_delimiter = multilabel_delimiter
        self.save = save

        #Shuffles the dataframe based on the input given
        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        
        #Creating a column called kfold and assigning dummy value to it.
        self.dataframe["kfold"] = -1
    
    def split(self):

        #Checking for the problem_type
        #For binary_classification and multiclass_classification
        if self.problem_type in ("binary_classification","multiclass_classification"):

            #Since it is classification the number of targets expected is 1
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")

            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()

            #The number of uniques values in the target is expected as at least 2
            if unique_values == 1:
                raise Exception("Only one unique value found for classification!")

            elif unique_values > 1: 
                kf = StratifiedKFold(n_splits=self.num_folds, 
                                    shuffle=False)

                #Creating folds
                for fold, (_, v_) in enumerate(kf.split(X=self.dataframe,
                                                        y= self.dataframe[target].values)):
                    self.dataframe.loc[v_, 'kfold'] = fold
        
        #For single_col_regression and multi_col_regression
        elif self.problem_type in ("single_col_regression", "multi_col_regression"):

            #For single column regression the number of targets should be 1
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")

            #For multi column regression there should be atleast 2 target columns
            if self.num_targets < 2 and self.problem_type == "multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            
            #Creating folds
            kf = KFold(n_splits=self.num_folds)
            for fold, (_, v_) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[v_, 'kfold'] = fold

        #For multilabel_classification
        elif self.problem_type == "multilabel_classification":

            #Since it is classification the number of targets expected is 1
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")

            #Since it is multi label classification we are splitting the targets with the delimiter
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            
            #Creating folds
            kf = StratifiedKFold(n_splits=self.num_folds)
            for fold, (_, v_) in enumerate(kf.split(X=self.dataframe, y= targets)):
                self.dataframe.loc[v_, 'kfold'] = fold
        
        #For holdout splits
        elif self.problem_type.startswith("holdout_"):
            #Getting the holdout percentage
            holdout_percentage = int(self.problem_type.split('_')[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe)-num_holdout_samples, 'kfold'] = 0
            self.dataframe.loc[len(self.dataframe)-num_holdout_samples:, 'kfold'] = 1


        else:
            raise Exception("Problem type not understood")
        
        if self.save == True:
            self.dataframe.to_csv(config.TRAINING_FOLD_DATA,index=False)
        
        return self.dataframe
