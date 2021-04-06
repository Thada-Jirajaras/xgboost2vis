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
    display(matrix.style.background_gradient())
    print()
    print()
```

    NA will show as -1
    ### class_one_prob=================================================================
    


<style  type="text/css" >
    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow0_col0 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow0_col1 {
            background-color:  #02395a;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow0_col2 {
            background-color:  #03466e;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow0_col3 {
            background-color:  #023a5b;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow1_col0 {
            background-color:  #023b5d;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow1_col1 {
            background-color:  #034369;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow1_col2 {
            background-color:  #034369;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow1_col3 {
            background-color:  #034d79;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow2_col0 {
            background-color:  #034369;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow2_col1 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow2_col2 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow2_col3 {
            background-color:  #03446a;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow3_col0 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow3_col1 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow3_col2 {
            background-color:  #034e7b;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow3_col3 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow4_col0 {
            background-color:  #034369;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow4_col1 {
            background-color:  #03456c;
            color:  #f1f1f1;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow4_col2 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow4_col3 {
            background-color:  #fff7fb;
            color:  #000000;
        }</style><table id="T_c4b53819_9707_11eb_9ec1_b46bfc620ecc" ><thead>    <tr>        <th class="index_name level0" >rules_y</th>        <th class="col_heading level0 col0" >['male < 1.0', 'sysBP >= 131.75']</th>        <th class="col_heading level0 col1" >['male >= 1.0', 'sysBP >= 131.75']</th>        <th class="col_heading level0 col2" >['totChol < 231.5', 'sysBP < 131.75']</th>        <th class="col_heading level0 col3" >['totChol >= 231.5', 'sysBP < 131.75']</th>    </tr>    <tr>        <th class="index_name level0" >rules_x</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_c4b53819_9707_11eb_9ec1_b46bfc620ecclevel0_row0" class="row_heading level0 row0" >['cigsPerDay < 16.5', 'age < 49.0']</th>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow0_col0" class="data row0 col0" >0.191083</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow0_col1" class="data row0 col1" >0.179487</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow0_col2" class="data row0 col2" >0.151329</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow0_col3" class="data row0 col3" >0.165957</td>
            </tr>
            <tr>
                        <th id="T_c4b53819_9707_11eb_9ec1_b46bfc620ecclevel0_row1" class="row_heading level0 row1" >['cigsPerDay >= 16.5', 'age < 49.0']</th>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow1_col0" class="data row1 col0" >0.173913</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow1_col1" class="data row1 col1" >0.136752</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow1_col2" class="data row1 col2" >0.162281</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow1_col3" class="data row1 col3" >0.081871</td>
            </tr>
            <tr>
                        <th id="T_c4b53819_9707_11eb_9ec1_b46bfc620ecclevel0_row2" class="row_heading level0 row2" >['male < 1.0', 'sysBP < 143.75', 'age >= 49.0']</th>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow2_col0" class="data row2 col0" >0.141304</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow2_col1" class="data row2 col1" >-1.000000</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow2_col2" class="data row2 col2" >0.214286</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow2_col3" class="data row2 col3" >0.125561</td>
            </tr>
            <tr>
                        <th id="T_c4b53819_9707_11eb_9ec1_b46bfc620ecclevel0_row3" class="row_heading level0 row3" >['male >= 1.0', 'sysBP < 143.75', 'age >= 49.0']</th>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow3_col0" class="data row3 col0" >-1.000000</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow3_col1" class="data row3 col1" >0.185185</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow3_col2" class="data row3 col2" >0.111111</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow3_col3" class="data row3 col3" >0.178161</td>
            </tr>
            <tr>
                        <th id="T_c4b53819_9707_11eb_9ec1_b46bfc620ecclevel0_row4" class="row_heading level0 row4" >['sysBP >= 143.75', 'age >= 49.0']</th>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow4_col0" class="data row4 col0" >0.144304</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow4_col1" class="data row4 col1" >0.125581</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow4_col2" class="data row4 col2" >-1.000000</td>
                        <td id="T_c4b53819_9707_11eb_9ec1_b46bfc620eccrow4_col3" class="data row4 col3" >-1.000000</td>
            </tr>
    </tbody></table>


    
    
    ### class_zero_count=================================================================
    


<style  type="text/css" >
    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow0_col0 {
            background-color:  #a5bddb;
            color:  #000000;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow0_col1 {
            background-color:  #6fa7ce;
            color:  #000000;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow0_col2 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow0_col3 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow1_col0 {
            background-color:  #eee9f3;
            color:  #000000;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow1_col1 {
            background-color:  #62a2cb;
            color:  #000000;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow1_col2 {
            background-color:  #84b0d3;
            color:  #000000;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow1_col3 {
            background-color:  #0567a1;
            color:  #f1f1f1;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow2_col0 {
            background-color:  #81aed2;
            color:  #000000;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow2_col1 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow2_col2 {
            background-color:  #dfddec;
            color:  #000000;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow2_col3 {
            background-color:  #02395a;
            color:  #f1f1f1;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow3_col0 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow3_col1 {
            background-color:  #4a98c5;
            color:  #000000;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow3_col2 {
            background-color:  #b0c2de;
            color:  #000000;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow3_col3 {
            background-color:  #0d75b3;
            color:  #f1f1f1;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow4_col0 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow4_col1 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow4_col2 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_c4badb43_9707_11eb_9646_b46bfc620eccrow4_col3 {
            background-color:  #fff7fb;
            color:  #000000;
        }</style><table id="T_c4badb43_9707_11eb_9646_b46bfc620ecc" ><thead>    <tr>        <th class="index_name level0" >rules_y</th>        <th class="col_heading level0 col0" >['male < 1.0', 'sysBP >= 131.75']</th>        <th class="col_heading level0 col1" >['male >= 1.0', 'sysBP >= 131.75']</th>        <th class="col_heading level0 col2" >['totChol < 231.5', 'sysBP < 131.75']</th>        <th class="col_heading level0 col3" >['totChol >= 231.5', 'sysBP < 131.75']</th>    </tr>    <tr>        <th class="index_name level0" >rules_x</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_c4badb43_9707_11eb_9646_b46bfc620ecclevel0_row0" class="row_heading level0 row0" >['cigsPerDay < 16.5', 'age < 49.0']</th>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow0_col0" class="data row0 col0" >127</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow0_col1" class="data row0 col1" >96</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow0_col2" class="data row0 col2" >415</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow0_col3" class="data row0 col3" >196</td>
            </tr>
            <tr>
                        <th id="T_c4badb43_9707_11eb_9646_b46bfc620ecclevel0_row1" class="row_heading level0 row1" >['cigsPerDay >= 16.5', 'age < 49.0']</th>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow1_col0" class="data row1 col0" >38</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow1_col1" class="data row1 col1" >101</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow1_col2" class="data row1 col2" >191</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow1_col3" class="data row1 col3" >157</td>
            </tr>
            <tr>
                        <th id="T_c4badb43_9707_11eb_9646_b46bfc620ecclevel0_row2" class="row_heading level0 row2" >['male < 1.0', 'sysBP < 143.75', 'age >= 49.0']</th>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow2_col0" class="data row2 col0" >158</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow2_col1" class="data row2 col1" >0</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow2_col2" class="data row2 col2" >77</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow2_col3" class="data row2 col3" >195</td>
            </tr>
            <tr>
                        <th id="T_c4badb43_9707_11eb_9646_b46bfc620ecclevel0_row3" class="row_heading level0 row3" >['male >= 1.0', 'sysBP < 143.75', 'age >= 49.0']</th>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow3_col0" class="data row3 col0" >0</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow3_col1" class="data row3 col1" >110</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow3_col2" class="data row3 col2" >144</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow3_col3" class="data row3 col3" >143</td>
            </tr>
            <tr>
                        <th id="T_c4badb43_9707_11eb_9646_b46bfc620ecclevel0_row4" class="row_heading level0 row4" >['sysBP >= 143.75', 'age >= 49.0']</th>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow4_col0" class="data row4 col0" >338</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow4_col1" class="data row4 col1" >188</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow4_col2" class="data row4 col2" >0</td>
                        <td id="T_c4badb43_9707_11eb_9646_b46bfc620eccrow4_col3" class="data row4 col3" >0</td>
            </tr>
    </tbody></table>


    
    
    ### class_one_count=================================================================
    


<style  type="text/css" >
    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow0_col0 {
            background-color:  #67a4cc;
            color:  #000000;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow0_col1 {
            background-color:  #056ba7;
            color:  #f1f1f1;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow0_col2 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow0_col3 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow1_col0 {
            background-color:  #e9e5f1;
            color:  #000000;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow1_col1 {
            background-color:  #4697c4;
            color:  #000000;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow1_col2 {
            background-color:  #73a9cf;
            color:  #000000;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow1_col3 {
            background-color:  #acc0dd;
            color:  #000000;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow2_col0 {
            background-color:  #86b0d3;
            color:  #000000;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow2_col1 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow2_col2 {
            background-color:  #c5cce3;
            color:  #000000;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow2_col3 {
            background-color:  #1278b4;
            color:  #f1f1f1;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow3_col0 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow3_col1 {
            background-color:  #034b76;
            color:  #f1f1f1;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow3_col2 {
            background-color:  #d2d2e7;
            color:  #000000;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow3_col3 {
            background-color:  #0568a3;
            color:  #f1f1f1;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow4_col0 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow4_col1 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow4_col2 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_c4be41b9_9707_11eb_a047_b46bfc620eccrow4_col3 {
            background-color:  #fff7fb;
            color:  #000000;
        }</style><table id="T_c4be41b9_9707_11eb_a047_b46bfc620ecc" ><thead>    <tr>        <th class="index_name level0" >rules_y</th>        <th class="col_heading level0 col0" >['male < 1.0', 'sysBP >= 131.75']</th>        <th class="col_heading level0 col1" >['male >= 1.0', 'sysBP >= 131.75']</th>        <th class="col_heading level0 col2" >['totChol < 231.5', 'sysBP < 131.75']</th>        <th class="col_heading level0 col3" >['totChol >= 231.5', 'sysBP < 131.75']</th>    </tr>    <tr>        <th class="index_name level0" >rules_x</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_c4be41b9_9707_11eb_a047_b46bfc620ecclevel0_row0" class="row_heading level0 row0" >['cigsPerDay < 16.5', 'age < 49.0']</th>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow0_col0" class="data row0 col0" >30</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow0_col1" class="data row0 col1" >21</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow0_col2" class="data row0 col2" >74</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow0_col3" class="data row0 col3" >39</td>
            </tr>
            <tr>
                        <th id="T_c4be41b9_9707_11eb_a047_b46bfc620ecclevel0_row1" class="row_heading level0 row1" >['cigsPerDay >= 16.5', 'age < 49.0']</th>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow1_col0" class="data row1 col0" >8</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow1_col1" class="data row1 col1" >16</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow1_col2" class="data row1 col2" >37</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow1_col3" class="data row1 col3" >14</td>
            </tr>
            <tr>
                        <th id="T_c4be41b9_9707_11eb_a047_b46bfc620ecclevel0_row2" class="row_heading level0 row2" >['male < 1.0', 'sysBP < 143.75', 'age >= 49.0']</th>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow2_col0" class="data row2 col0" >26</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow2_col1" class="data row2 col1" >0</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow2_col2" class="data row2 col2" >21</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow2_col3" class="data row2 col3" >28</td>
            </tr>
            <tr>
                        <th id="T_c4be41b9_9707_11eb_a047_b46bfc620ecclevel0_row3" class="row_heading level0 row3" >['male >= 1.0', 'sysBP < 143.75', 'age >= 49.0']</th>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow3_col0" class="data row3 col0" >0</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow3_col1" class="data row3 col1" >25</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow3_col2" class="data row3 col2" >18</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow3_col3" class="data row3 col3" >31</td>
            </tr>
            <tr>
                        <th id="T_c4be41b9_9707_11eb_a047_b46bfc620ecclevel0_row4" class="row_heading level0 row4" >['sysBP >= 143.75', 'age >= 49.0']</th>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow4_col0" class="data row4 col0" >57</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow4_col1" class="data row4 col1" >27</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow4_col2" class="data row4 col2" >0</td>
                        <td id="T_c4be41b9_9707_11eb_a047_b46bfc620eccrow4_col3" class="data row4 col3" >0</td>
            </tr>
    </tbody></table>


    
    
    
