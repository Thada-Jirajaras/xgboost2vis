import xgboost as xgb
import pandas as pd
import joblib
import os



def model_fn(model_path = os.path.join('model', 'model.pkl')):
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return(model)

def predict(model, x, y):
    
    # predict samples 
    output = pd.DataFrame([[ f'{i}-{p}' for i,p in enumerate(pred)] for pred in model.get_booster().predict(xgb.DMatrix(x), pred_leaf = True).tolist()])
    output['y'] = y
    output['sample_index'] = range(output.shape[0])
    output = output.rename(columns = {0: 'leaf_index_x', 1: 'leaf_index_y'})
    output = output[['sample_index', 'leaf_index_x', 'leaf_index_y', 'y']]
    l = pd.merge(output, model.rule_df, left_on = 'leaf_index_x', right_on = 'leaf_index', how = 'left')[['sample_index', 'leaf_index_x', 'leaf_index_y', 'rules', 'y']]
    output = pd.merge(l, model.rule_df, left_on = 'leaf_index_y', right_on = 'leaf_index', how = 'left')[['sample_index', 'leaf_index_x', 'leaf_index_y', 'rules_x', 'rules_y', 'y']]
    output['rules_x'] = output.rules_x.astype(str)
    output['rules_y'] = output.rules_y.astype(str)
    output = output.groupby(['y', 'rules_x', 'rules_y']).sample_index.nunique().reset_index()
    output['class_count'] = output.apply(lambda x: ('c' + str(x['y']), x['sample_index']), axis = 1)
    output = output.groupby(['rules_x', 'rules_y'], as_index = False).class_count.agg(lambda x: dict(list(x)) )
    output = pd.pivot(output, index = 'rules_x', columns = 'rules_y')

    return(output)