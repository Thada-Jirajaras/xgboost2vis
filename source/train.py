import os
import argparse
import joblib
import pandas as pd
from source.model import XGBoost2Vis


def model_fn(model_path = os.path.join('model', 'model.pkl')):
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return(model)
        
def train(model, 
          xtrain, ytrain,
         model_path = os.path.join('model', 'model.pkl')):
    
    
    
    # fit model
    model.fit(xtrain, ytrain)
    
    # extract rules from fitted model
    dataset = model._Booster.trees_to_dataframe()
    dataset = pd.melt(dataset, id_vars = ['ID', "Feature", "Split"], value_vars= ["Yes", "No", "Missing"], 
            var_name = 'cat', value_name = 'to')
    leaves = dataset.loc[dataset.Feature == 'Leaf', ['ID']] #.join(dataset, how = 'left', on =)
    nodes = dataset[['ID', 'to', 'Feature', 'Split']].dropna()
    k = pd.merge(leaves, nodes, how = 'left', left_on='ID', right_on = 'to')[[ 'ID_y', 'Feature', 'Split']]
    k = k.rename(columns = {'ID_y': 'ID'})
    k['index_node'] = k['ID'].copy()
    k['Rule_1'] =  list(zip(k['Feature'], k['Split']))
    k = k[['index_node', 'ID', 'Rule_1']].drop_duplicates()
    do_more = (k['ID'].str.split('-').map(lambda x: x[1]).astype(int) > 0)
    keep_k = []
    while do_more.any():
        k = pd.merge(k, nodes, how = 'left', left_on='ID', right_on = 'to').drop(columns=['ID_x', 'to']).rename(columns = {'ID_y': 'ID'}).drop_duplicates()
        k['Rule_2'] =  list(zip(k['Feature'], k['Split']))
        del k['Feature'] 
        del k['Split'] 
        k = k.drop_duplicates()
        keep_k.append(k[(k['ID'].str.split('-').map(lambda x: x[1]).astype(int) == 0)])
        do_more = (k['ID'].str.split('-').map(lambda x: x[1]).astype(int) > 0)
        k = k[do_more]   
    keep_k = pd.concat(keep_k)
    model.rule_df = keep_k.drop_duplicates()
    model.rule_df 
    
    # save model
    with open(model_path, 'wb') as f:
        joblib.dump(model, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--integers', type = int, default = -1)
    args = parser.parse_args()
    
    model = XGBoost2Vis()
    model.fit()
    
    
    