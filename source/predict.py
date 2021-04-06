import xgboost as xgb
import pandas as pd
import joblib
import os



def model_fn(model_path = os.path.join('model', 'model.pkl')):
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return(model)

def predict(model, x):
    
    output = pd.DataFrame([[ f'{i}-{p}' for i,p in enumerate(pred)] for pred in model.get_booster().predict(xgb.DMatrix(x), pred_leaf = True).tolist()])
    output['sample_index'] = range(output.shape[0])
    output = pd.melt(output, id_vars = 'sample_index', value_name = 'leaf_index', var_name = 'rule_number')
    output = pd.merge(output, model.rule_df, on = 'leaf_index', how = 'left')
    output = output[['sample_index', 'Tree', 'leaf_index', 'rules']]
    return(output)