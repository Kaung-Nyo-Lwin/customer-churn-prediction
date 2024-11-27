import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.ensemble import GradientBoostingClassifier
import preprocessing


# param_grid = {
#     "model__loss":["log_loss", "exponential"],
#     "model__learning_rate": [0.1,0.01,0.001],
#     "model__min_samples_split": np.linspace(0.1, 0.5, 12),
#     "model__min_samples_leaf": np.linspace(0.1, 0.5, 12),
#     "model__max_depth":[3, 5, 10, None],
#     "model__max_features":["log2","sqrt"],
#     "model__criterion": ["friedman_mse",  "mae"],
#     "model__subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
#     "model__n_estimators":[50, 60, 70, 80, 90, 100, 110,120, 130,140, 150]
#     }

def search(param_grid,split=5,model=GradientBoostingClassifier,upsample=False):
    X_train, _, y_train, _ = preprocessing.prepare()
    if upsample == True:
        X_train,y_train = preprocessing.up_sample(X_train,y_train)
    kfold = KFold(n_splits=split, shuffle=True)
    params = {}
    pipeline = preprocessing.preprocess_pipeline()
    for param,values in param_grid.items():
        param_values = {}
        param_values[param] = values

        model_pipe = Pipeline([('preprocess',pipeline),('model',model)])

        grid = GridSearchCV(estimator = model_pipe, 
                            param_grid = param_values, 
                            cv = kfold, 
                            n_jobs = -1, 
                            return_train_score=True, 
                            refit=True,
                            scoring='recall_weighted')

    #    Fit your grid_search
        grid.fit(X_train, y_train);  #fit means start looping all the possible parameters

        params[param.replace('model__','')] = grid.best_params_[param]
        
    return params
    
