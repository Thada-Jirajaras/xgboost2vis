# xgboost2vis

# Read data


```python
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

iris = load_iris()
x, y = iris.data, iris.target
y =  (y> 1).astype(int)
```

# Train XGBoost2vis model


```python
from source.model import XGBoost2Vis
from source.train import train, model_fn
import xgboost as xgb
from xgboost import plot_tree
model = XGBoost2Vis()
train(model, x, y)
```

# Use model to visualise data by group and class


```python
from source.predict import predict, model_fn
model = model_fn()
result = predict(model, x, y)
result 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">class_count</th>
    </tr>
    <tr>
      <th>rules_y</th>
      <th>['f0 &lt; 6.05', 'f3 &gt;= 1.75']</th>
      <th>['f0 &gt;= 6.05', 'f3 &gt;= 1.75']</th>
      <th>['f2 &lt; 4.95', 'f3 &lt; 1.75']</th>
      <th>['f2 &gt;= 4.95', 'f3 &lt; 1.75']</th>
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
      <th>['f2 &lt; 4.85', 'f3 &gt;= 1.65']</th>
      <td>{'c0': 1, 'c1': 1}</td>
      <td>{'c1': 1}</td>
      <td>{'c1': 1}</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>['f2 &lt; 4.95', 'f3 &lt; 1.65']</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>{'c0': 97}</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>['f2 &gt;= 4.85', 'f3 &gt;= 1.65']</th>
      <td>{'c1': 6}</td>
      <td>{'c1': 37}</td>
      <td>NaN</td>
      <td>{'c0': 1}</td>
    </tr>
    <tr>
      <th>['f2 &gt;= 4.95', 'f3 &lt; 1.65']</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>{'c0': 1, 'c1': 4}</td>
    </tr>
  </tbody>
</table>
</div>


