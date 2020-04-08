

```python
import pandas as pd
import numpy as np
```


```python
# get the dataset
# ! curl https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv -o 'temps.csv'
```


```python
ts = pd.read_csv('temps.csv')
ts.head()
```




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
      <th></th>
      <th>Date</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1981-01-01</td>
      <td>20.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1981-01-02</td>
      <td>17.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1981-01-03</td>
      <td>18.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981-01-04</td>
      <td>14.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981-01-05</td>
      <td>15.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What is the first thing we do with a time series dataset

```


```python
# What is the shape
ts.shape
```




    (3650, 1)




```python
ts.isna().sum()
```




    Temp    0
    dtype: int64




```python

```
