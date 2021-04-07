```python
%load_ext autoreload
%autoreload 2
```

# Read data


```python
import pandas as  pd
import os
dataset = pd.read_csv(os.path.join('data', 'framingham.csv')).dropna()
y_col  = 'TenYearCHD'
x, y = dataset.drop(columns = [y_col]), dataset[y_col]
num_sample = x.shape[0]
scale_pos_weight = ((y.value_counts().max()/y.value_counts())**0.75)[1]
```

# Train XGBoost2vis model


```python
from source.model import XGBoost2Vis
from source.train import train
import xgboost as xgb
from xgboost import plot_tree
model = XGBoost2Vis(scale_pos_weight = scale_pos_weight, 
                   min_child_weight = int(num_sample*0.05))
train(model, x, y)
```

# Use model to visualise data by group and class


```python
from source.predict import predict, model_fn
model = model_fn()
class_zero_count, class_one_count, class_one_prob = predict(model, x, y)

print('NA will show as -0.00001')
for name, matrix in {'class_one_prob': class_one_prob.fillna(-0.00001), 'class_zero_count': class_zero_count.fillna(0).astype(int), 
                     'class_one_count': class_one_count.fillna(0).astype(int)}.items():
    print(f'### {name}=================================================================')
    display(matrix.style.background_gradient(axis = None))
    print()
    print()
```
![result](image/result.png)
