import xgboost as xgb



class XGBoost2Vis(xgb.XGBClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, n_estimators = 2, 
                         eta = 1, max_depth = 3)
        