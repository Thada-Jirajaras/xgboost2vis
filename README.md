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

print('NA will show as -1')
for name, matrix in {'class_one_prob': class_one_prob.fillna(-1), 'class_zero_count': class_zero_count.fillna(0).astype(int), 
                     'class_one_count': class_one_count.fillna(0).astype(int)}.items():
    print(f'### {name}=================================================================')
    display(matrix)#.style.background_gradient())
    print()
    print()
```

    NA will show as -1
    ### class_one_prob=================================================================
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>rules_y</th>
      <th>['male &lt; 1.0', 'sysBP &gt;= 131.75']</th>
      <th>['male &gt;= 1.0', 'sysBP &gt;= 131.75']</th>
      <th>['totChol &lt; 231.5', 'sysBP &lt; 131.75']</th>
      <th>['totChol &gt;= 231.5', 'sysBP &lt; 131.75']</th>
    </tr>
    <tr>
      <th>rules_x</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>['cigsPerDay &lt; 16.5', 'age &lt; 49.0']</th>
      <td>0.191083</td>
      <td>0.179487</td>
      <td>0.151329</td>
      <td>0.165957</td>
    </tr>
    <tr>
      <th>['cigsPerDay &gt;= 16.5', 'age &lt; 49.0']</th>
      <td>0.173913</td>
      <td>0.136752</td>
      <td>0.162281</td>
      <td>0.081871</td>
    </tr>
    <tr>
      <th>['male &lt; 1.0', 'sysBP &lt; 143.75', 'age &gt;= 49.0']</th>
      <td>0.141304</td>
      <td>-1.000000</td>
      <td>0.214286</td>
      <td>0.125561</td>
    </tr>
    <tr>
      <th>['male &gt;= 1.0', 'sysBP &lt; 143.75', 'age &gt;= 49.0']</th>
      <td>-1.000000</td>
      <td>0.185185</td>
      <td>0.111111</td>
      <td>0.178161</td>
    </tr>
    <tr>
      <th>['sysBP &gt;= 143.75', 'age &gt;= 49.0']</th>
      <td>0.144304</td>
      <td>0.125581</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    ### class_zero_count=================================================================
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>rules_y</th>
      <th>['male &lt; 1.0', 'sysBP &gt;= 131.75']</th>
      <th>['male &gt;= 1.0', 'sysBP &gt;= 131.75']</th>
      <th>['totChol &lt; 231.5', 'sysBP &lt; 131.75']</th>
      <th>['totChol &gt;= 231.5', 'sysBP &lt; 131.75']</th>
    </tr>
    <tr>
      <th>rules_x</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>['cigsPerDay &lt; 16.5', 'age &lt; 49.0']</th>
      <td>127</td>
      <td>96</td>
      <td>415</td>
      <td>196</td>
    </tr>
    <tr>
      <th>['cigsPerDay &gt;= 16.5', 'age &lt; 49.0']</th>
      <td>38</td>
      <td>101</td>
      <td>191</td>
      <td>157</td>
    </tr>
    <tr>
      <th>['male &lt; 1.0', 'sysBP &lt; 143.75', 'age &gt;= 49.0']</th>
      <td>158</td>
      <td>0</td>
      <td>77</td>
      <td>195</td>
    </tr>
    <tr>
      <th>['male &gt;= 1.0', 'sysBP &lt; 143.75', 'age &gt;= 49.0']</th>
      <td>0</td>
      <td>110</td>
      <td>144</td>
      <td>143</td>
    </tr>
    <tr>
      <th>['sysBP &gt;= 143.75', 'age &gt;= 49.0']</th>
      <td>338</td>
      <td>188</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    ### class_one_count=================================================================
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>rules_y</th>
      <th>['male &lt; 1.0', 'sysBP &gt;= 131.75']</th>
      <th>['male &gt;= 1.0', 'sysBP &gt;= 131.75']</th>
      <th>['totChol &lt; 231.5', 'sysBP &lt; 131.75']</th>
      <th>['totChol &gt;= 231.5', 'sysBP &lt; 131.75']</th>
    </tr>
    <tr>
      <th>rules_x</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>['cigsPerDay &lt; 16.5', 'age &lt; 49.0']</th>
      <td>30</td>
      <td>21</td>
      <td>74</td>
      <td>39</td>
    </tr>
    <tr>
      <th>['cigsPerDay &gt;= 16.5', 'age &lt; 49.0']</th>
      <td>8</td>
      <td>16</td>
      <td>37</td>
      <td>14</td>
    </tr>
    <tr>
      <th>['male &lt; 1.0', 'sysBP &lt; 143.75', 'age &gt;= 49.0']</th>
      <td>26</td>
      <td>0</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>['male &gt;= 1.0', 'sysBP &lt; 143.75', 'age &gt;= 49.0']</th>
      <td>0</td>
      <td>25</td>
      <td>18</td>
      <td>31</td>
    </tr>
    <tr>
      <th>['sysBP &gt;= 143.75', 'age &gt;= 49.0']</th>
      <td>57</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    
