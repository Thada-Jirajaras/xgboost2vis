import os
import argparse
import joblib
import pandas as pd
from pathlib import Path
from source.model import XGBoost2Vis


def model_fn(model_path = os.path.join('model', 'model.pkl')):
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return(model)
        
def train(model, 
          xtrain, ytrain,
          model_path = 'model',
          model_file = 'model.pkl'):

    
    
    # fit model
    model.fit(xtrain, ytrain)
    
    # extract rules from fitted model
    ruledf = model._Booster.trees_to_dataframe()
    ruledf['missing_with_yes'] = ruledf['Missing'] == ruledf['Yes']
    ruledf['missing_with_no'] =  ruledf['Missing'] == ruledf['No']
    ruledf = pd.melt(ruledf, id_vars = ['ID', "Tree", "Feature", "Split", "missing_with_yes", "missing_with_no"], value_vars= ["Yes", "No"], 
                var_name = 'cat', value_name = 'to')
    ruledf['Split'] = ruledf['Split'].map(lambda x: round(x, 2))
    ruledf['rule'] = ruledf[['Feature','Split', 
                             'cat', 'missing_with_yes',
                            'missing_with_no']].apply(lambda x: f'{x["Feature"]} < {x["Split"]}' + ('or missing' if x['missing_with_yes'] else '') if x['cat'] in set(['Yes', 'Missing']) 
                                else f'{x["Feature"]} >= {x["Split"]}'  + ('or missing' if x['missing_with_no']  else ''), axis=1).drop(columns = ['missing_with_yes', 'missing_with_no'])
    leaves = ruledf.loc[ruledf.Feature == 'Leaf', ["Tree", 'ID']].drop_duplicates().rename(columns = {'ID': 'leaf_index'})
    nodes = ruledf[['ID', 'to', 'rule']].dropna().drop_duplicates()
    def rule_for_each_leave(leaf_index, nodes):
        queryID = leaf_index
        a = nodes[nodes.to==queryID]
        ruleresult = [a]
        while a.shape[0] > 0:
            a = nodes[nodes.to.isin(a.ID)]
            ruleresult.append(a)
        return(pd.concat(ruleresult).rule.tolist()[::-1])
    leaves['rules'] = leaves.leaf_index.map(lambda x: rule_for_each_leave(x, nodes))
    model.rule_df = leaves
    
    # save model
    p = Path(model_path) 
    p.mkdir(exist_ok=True)
    with open(os.path.join(model_path, model_file), 'wb') as f:  
        joblib.dump(model, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--integers', type = int, default = -1)
    args = parser.parse_args()
    
    model = XGBoost2Vis()
    model.fit()
    
    
    