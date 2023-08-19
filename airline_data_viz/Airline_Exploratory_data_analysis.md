# Enhancing Airline Passenger Satisfaction: Data-driven Insights for Superior Travel Experiences.

## Getting you up to speed :

### If you've ever wondered what makes airline passengers truly happy, you're in for a treat. 
### This project is all about digging into a bunch of data about how travelers feel about their flights. My plan? 
### To use some cool tech skills to figure out what things matter most to passengers, from looking at stuff like how long flights are, to checking if delays annoy folks, I'll be using my data analysis superpowers to unravel the secrets of happy flyers. 
### So buckle up, as we dive into this data adventure to help airlines make your next trip even more awesome!

#### First, let's import the required libraries 



```python
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

```


```python
pd.read_csv("Datasets\Airline+Passenger+Satisfaction/airline_passenger_satisfaction.csv")

airline_data = pd.read_csv("Datasets\Airline+Passenger+Satisfaction/airline_passenger_satisfaction.csv")
airline_data
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
      <th>ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Customer Type</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Departure Delay</th>
      <th>Arrival Delay</th>
      <th>Departure and Arrival Time Convenience</th>
      <th>...</th>
      <th>On-board Service</th>
      <th>Seat Comfort</th>
      <th>Leg Room Service</th>
      <th>Cleanliness</th>
      <th>Food and Drink</th>
      <th>In-flight Service</th>
      <th>In-flight Wifi Service</th>
      <th>In-flight Entertainment</th>
      <th>Baggage Handling</th>
      <th>Satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>48</td>
      <td>First-time</td>
      <td>Business</td>
      <td>Business</td>
      <td>821</td>
      <td>2</td>
      <td>5.0</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Female</td>
      <td>35</td>
      <td>Returning</td>
      <td>Business</td>
      <td>Business</td>
      <td>821</td>
      <td>26</td>
      <td>39.0</td>
      <td>2</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Male</td>
      <td>41</td>
      <td>Returning</td>
      <td>Business</td>
      <td>Business</td>
      <td>853</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>...</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Male</td>
      <td>50</td>
      <td>Returning</td>
      <td>Business</td>
      <td>Business</td>
      <td>1905</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>49</td>
      <td>Returning</td>
      <td>Business</td>
      <td>Business</td>
      <td>3470</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>129875</th>
      <td>129876</td>
      <td>Male</td>
      <td>28</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>447</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
    <tr>
      <th>129876</th>
      <td>129877</td>
      <td>Male</td>
      <td>41</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>308</td>
      <td>0</td>
      <td>0.0</td>
      <td>5</td>
      <td>...</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
    <tr>
      <th>129877</th>
      <td>129878</td>
      <td>Male</td>
      <td>42</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>337</td>
      <td>6</td>
      <td>14.0</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
    <tr>
      <th>129878</th>
      <td>129879</td>
      <td>Male</td>
      <td>50</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>337</td>
      <td>31</td>
      <td>22.0</td>
      <td>4</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>129879</th>
      <td>129880</td>
      <td>Female</td>
      <td>20</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>337</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
  </tbody>
</table>
<p>129880 rows × 24 columns</p>
</div>



#### Dealing with missing values 


```python
airline_data.dropna(inplace=True)
airline_data
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
      <th>ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Customer Type</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Departure Delay</th>
      <th>Arrival Delay</th>
      <th>Departure and Arrival Time Convenience</th>
      <th>...</th>
      <th>On-board Service</th>
      <th>Seat Comfort</th>
      <th>Leg Room Service</th>
      <th>Cleanliness</th>
      <th>Food and Drink</th>
      <th>In-flight Service</th>
      <th>In-flight Wifi Service</th>
      <th>In-flight Entertainment</th>
      <th>Baggage Handling</th>
      <th>Satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>48</td>
      <td>First-time</td>
      <td>Business</td>
      <td>Business</td>
      <td>821</td>
      <td>2</td>
      <td>5.0</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Female</td>
      <td>35</td>
      <td>Returning</td>
      <td>Business</td>
      <td>Business</td>
      <td>821</td>
      <td>26</td>
      <td>39.0</td>
      <td>2</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Male</td>
      <td>41</td>
      <td>Returning</td>
      <td>Business</td>
      <td>Business</td>
      <td>853</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>...</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Male</td>
      <td>50</td>
      <td>Returning</td>
      <td>Business</td>
      <td>Business</td>
      <td>1905</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>49</td>
      <td>Returning</td>
      <td>Business</td>
      <td>Business</td>
      <td>3470</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>129875</th>
      <td>129876</td>
      <td>Male</td>
      <td>28</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>447</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
    <tr>
      <th>129876</th>
      <td>129877</td>
      <td>Male</td>
      <td>41</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>308</td>
      <td>0</td>
      <td>0.0</td>
      <td>5</td>
      <td>...</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
    <tr>
      <th>129877</th>
      <td>129878</td>
      <td>Male</td>
      <td>42</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>337</td>
      <td>6</td>
      <td>14.0</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
    <tr>
      <th>129878</th>
      <td>129879</td>
      <td>Male</td>
      <td>50</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>337</td>
      <td>31</td>
      <td>22.0</td>
      <td>4</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>129879</th>
      <td>129880</td>
      <td>Female</td>
      <td>20</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>337</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
  </tbody>
</table>
<p>129487 rows × 24 columns</p>
</div>




```python
airline_data.isnull().sum()
```




    ID                                        0
    Gender                                    0
    Age                                       0
    Customer Type                             0
    Type of Travel                            0
    Class                                     0
    Flight Distance                           0
    Departure Delay                           0
    Arrival Delay                             0
    Departure and Arrival Time Convenience    0
    Ease of Online Booking                    0
    Check-in Service                          0
    Online Boarding                           0
    Gate Location                             0
    On-board Service                          0
    Seat Comfort                              0
    Leg Room Service                          0
    Cleanliness                               0
    Food and Drink                            0
    In-flight Service                         0
    In-flight Wifi Service                    0
    In-flight Entertainment                   0
    Baggage Handling                          0
    Satisfaction                              0
    dtype: int64




```python
airline_data.tail()
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
      <th>ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Customer Type</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Departure Delay</th>
      <th>Arrival Delay</th>
      <th>Departure and Arrival Time Convenience</th>
      <th>...</th>
      <th>On-board Service</th>
      <th>Seat Comfort</th>
      <th>Leg Room Service</th>
      <th>Cleanliness</th>
      <th>Food and Drink</th>
      <th>In-flight Service</th>
      <th>In-flight Wifi Service</th>
      <th>In-flight Entertainment</th>
      <th>Baggage Handling</th>
      <th>Satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>129875</th>
      <td>129876</td>
      <td>Male</td>
      <td>28</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>447</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
    <tr>
      <th>129876</th>
      <td>129877</td>
      <td>Male</td>
      <td>41</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>308</td>
      <td>0</td>
      <td>0.0</td>
      <td>5</td>
      <td>...</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
    <tr>
      <th>129877</th>
      <td>129878</td>
      <td>Male</td>
      <td>42</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>337</td>
      <td>6</td>
      <td>14.0</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
    <tr>
      <th>129878</th>
      <td>129879</td>
      <td>Male</td>
      <td>50</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>337</td>
      <td>31</td>
      <td>22.0</td>
      <td>4</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>129879</th>
      <td>129880</td>
      <td>Female</td>
      <td>20</td>
      <td>Returning</td>
      <td>Personal</td>
      <td>Economy Plus</td>
      <td>337</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>Neutral or Dissatisfied</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
airline_data.dtypes
```




    ID                                          int64
    Gender                                     object
    Age                                         int64
    Customer Type                              object
    Type of Travel                             object
    Class                                      object
    Flight Distance                             int64
    Departure Delay                             int64
    Arrival Delay                             float64
    Departure and Arrival Time Convenience      int64
    Ease of Online Booking                      int64
    Check-in Service                            int64
    Online Boarding                             int64
    Gate Location                               int64
    On-board Service                            int64
    Seat Comfort                                int64
    Leg Room Service                            int64
    Cleanliness                                 int64
    Food and Drink                              int64
    In-flight Service                           int64
    In-flight Wifi Service                      int64
    In-flight Entertainment                     int64
    Baggage Handling                            int64
    Satisfaction                               object
    dtype: object



#### Now that the data looks clean, it is indeed ready to answer analytical questions

### How is passenger satisfaction distributed across different gender groups?

#### We can answer this by analyzing the satisfaction levels for female and male passengers and see if there are any notable differences.


```python
# Group data by Gender and Satisfaction, and count occurrences

passenger_satisfaction = airline_data.groupby(['Gender', 'Satisfaction']).size().unstack()

passenger_satisfaction
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
      <th>Satisfaction</th>
      <th>Neutral or Dissatisfied</th>
      <th>Satisfied</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>37524</td>
      <td>28179</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>35701</td>
      <td>28083</td>
    </tr>
  </tbody>
</table>
</div>



### Does age affect overall passenger satisfaction?

#### We can group passengers by age ranges and compare their overall satisfaction levels to see if there's a correlation.


```python
# Define age ranges
age_bins = [0, 18, 30, 50, 100]
age_labels = ['0-18', '19-30', '31-50', '51+']

# Add a new column 'age_range' to the DataFrame based on age bins
airline_data['age_range'] = pd.cut(airline_data['Age'], bins=age_bins, labels=age_labels, right=False)

airline_data['Satisfied'] = airline_data['Satisfaction'] == 'Satisfied'
airline_data['Unsatisfied'] = airline_data['Satisfaction'] == 'Neutral or Dissatisfied'

# Group by age_range and calculate the count of 'Satisfied' occurrences for each group
age_satisfaction = airline_data.groupby('age_range')[['Satisfied', 'Unsatisfied']].sum()
age_satisfaction
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
      <th>Satisfied</th>
      <th>Unsatisfied</th>
    </tr>
    <tr>
      <th>age_range</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0-18</th>
      <td>1641</td>
      <td>8170</td>
    </tr>
    <tr>
      <th>19-30</th>
      <td>9914</td>
      <td>18514</td>
    </tr>
    <tr>
      <th>31-50</th>
      <td>27727</td>
      <td>27451</td>
    </tr>
    <tr>
      <th>51+</th>
      <td>16980</td>
      <td>19090</td>
    </tr>
  </tbody>
</table>
</div>



### Are returning customers generally more satisfied than first-time customers?

#### We can compare the satisfaction levels of returning and first-time customers to understand if loyalty affects satisfaction.


```python
# Classifying first-time customers and returning ones.
returning_customers = airline_data[airline_data['Customer Type'] == 'Returning']
first_time_customers = airline_data[airline_data['Customer Type'] == 'First-time']

# Calculate average satisfaction levels for returning and first-time customers
avg_satisfaction_returning = returning_customers['Satisfied'].mean()
avg_satisfaction_first_time = first_time_customers['Satisfied'].mean()

print("Average satisfaction level for returning customers:", avg_satisfaction_returning.round(2))
print("Average satisfaction level for first-time customers:", avg_satisfaction_first_time.round(2))
```

    Average satisfaction level for returning customers: 0.48
    Average satisfaction level for first-time customers: 0.24
    


```python
# Perform a t-test to check if the difference is statistically significant
from scipy.stats import ttest_ind

t_statistic, p_value = ttest_ind(returning_customers['Satisfied'], first_time_customers['Satisfied'])

if p_value < 0.05:
    print("The difference in satisfaction levels is statistically significant.")
else:
    print("The difference in satisfaction levels is not statistically significant.")
```

    The difference in satisfaction levels is statistically significant.
    

### Is there a difference in satisfaction levels between business and personal travelers?

####  We will analyze whether passengers on business trips have different satisfaction levels compared to those traveling for personal reasons.


```python
# Group data by 'Travel_Type' and calculate the mean satisfaction for each group

satisfaction_by_type = airline_data.groupby('Type of Travel')[['Satisfied','Unsatisfied']].mean().round(2)

satisfaction_by_type

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
      <th>Satisfied</th>
      <th>Unsatisfied</th>
    </tr>
    <tr>
      <th>Type of Travel</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Business</th>
      <td>0.58</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>Personal</th>
      <td>0.10</td>
      <td>0.90</td>
    </tr>
  </tbody>
</table>
</div>



### Which travel class has the highest satisfaction rating?

#### We will investigate which travel class (e.g., Economy, Business, First) receives the highest satisfaction ratings.



```python
# Calculate the average satisfaction rating for each travel class

average_satisfaction_by_class = airline_data.groupby('Class')['Satisfied'].mean()

# Find the travel class with the highest average satisfaction

highest_satisfaction_class = average_satisfaction_by_class.idxmax()
highest_satisfaction_value = average_satisfaction_by_class.round(2).max()

# Create a new DataFrame to display the result

satifaction_by_class = pd.DataFrame({
    'Travel Class': [highest_satisfaction_class],
    'Highest Satisfaction Rating': [highest_satisfaction_value]
})

satifaction_by_class
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
      <th>Travel Class</th>
      <th>Highest Satisfaction Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Business</td>
      <td>0.69</td>
    </tr>
  </tbody>
</table>
</div>



### Do longer flight distances have a direct effect with lower satisfaction levels?

#### We could determine if passengers on longer flights tend to have lower satisfaction scores compared to those on shorter flights.


```python
# Convert 'Flight Distance' column to numeric type

dtype={'Flight Distance': float}

# Calculate the average satisfaction score for different distance ranges

distance_bins = [0, 500, 1000, 1500, 2000]  
distance_labels = ['<500', '500-1000', '1000-1500', '1500+']

airline_data['Flight Distance'] = pd.cut(airline_data['Flight Distance'], bins=distance_bins, labels=distance_labels)
average_satisfaction_by_distance = airline_data.groupby('Flight Distance')[['Satisfied']].mean()

average_satisfaction_by_distance
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
      <th>Satisfied</th>
    </tr>
    <tr>
      <th>Flight Distance</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>&lt;500</th>
      <td>0.335657</td>
    </tr>
    <tr>
      <th>500-1000</th>
      <td>0.324809</td>
    </tr>
    <tr>
      <th>1000-1500</th>
      <td>0.363494</td>
    </tr>
    <tr>
      <th>1500+</th>
      <td>0.581687</td>
    </tr>
  </tbody>
</table>
</div>



### How do different aspects of the travel experience (e.g., cleanliness, food, entertainment) contribute to overall satisfaction?

####   We will break down the satisfaction scores for various aspects of the flight to see which factors have the most impact on overall satisfaction.


```python
# Creating a new dataframe that only displays data about the travel experience 

# List of columns selected
selected_columns = [
    'Cleanliness',
    'Food and Drink',
    'In-flight Entertainment',
    'Seat Comfort',
    'Leg Room Service',
    'In-flight Service',
    'In-flight Wifi Service',
    'Online Boarding',
    'Ease of Online Booking',
    'Departure and Arrival Time Convenience',
    'On-board Service',
     'Check-in Service',
    'Baggage Handling',
    'Gate Location'
    
]

travel_experience = airline_data[selected_columns]

# Group by Flight Type and calculate mean for each aspect

travel_experience_by_class = travel_experience.groupby(airline_data['Class']).mean()

# Calculate the overall satisfaction for each row (axis=1 means calculate across columns)

travel_experience_by_class['Overall Satisfaction'] = travel_experience_by_class.mean(axis=1)

# Sort by the overall satisfaction in descending order

travel_experience_by_class = travel_experience_by_class.sort_values(
    by=[
    'Cleanliness',
    'Food and Drink',
    'In-flight Entertainment',
    'Seat Comfort',
    'Leg Room Service',
    'In-flight Service',
    'In-flight Wifi Service',
    'Online Boarding',
    'Ease of Online Booking',
    'Departure and Arrival Time Convenience',
    'On-board Service',
     'Check-in Service',
    'Baggage Handling',
    'Gate Location'
    
],
    ascending=False
)

# Rearrange columns to have 'Overall Satisfaction' as the first column

column_order = ['Overall Satisfaction'] + selected_columns
travel_experience_by_class = travel_experience_by_class[column_order]

travel_experience_by_class
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
      <th>Overall Satisfaction</th>
      <th>Cleanliness</th>
      <th>Food and Drink</th>
      <th>In-flight Entertainment</th>
      <th>Seat Comfort</th>
      <th>Leg Room Service</th>
      <th>In-flight Service</th>
      <th>In-flight Wifi Service</th>
      <th>Online Boarding</th>
      <th>Ease of Online Booking</th>
      <th>Departure and Arrival Time Convenience</th>
      <th>On-board Service</th>
      <th>Check-in Service</th>
      <th>Baggage Handling</th>
      <th>Gate Location</th>
    </tr>
    <tr>
      <th>Class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Business</th>
      <td>3.432669</td>
      <td>3.481933</td>
      <td>3.329795</td>
      <td>3.639313</td>
      <td>3.763704</td>
      <td>3.646169</td>
      <td>3.846007</td>
      <td>2.775657</td>
      <td>3.719035</td>
      <td>2.915373</td>
      <td>2.907582</td>
      <td>3.682529</td>
      <td>3.520745</td>
      <td>3.844539</td>
      <td>2.984981</td>
    </tr>
    <tr>
      <th>Economy Plus</th>
      <td>3.060029</td>
      <td>3.118017</td>
      <td>3.110554</td>
      <td>3.120469</td>
      <td>3.168763</td>
      <td>3.056610</td>
      <td>3.382303</td>
      <td>2.755864</td>
      <td>2.886247</td>
      <td>2.662793</td>
      <td>3.209382</td>
      <td>3.034755</td>
      <td>3.014606</td>
      <td>3.351812</td>
      <td>2.968230</td>
    </tr>
    <tr>
      <th>Economy</th>
      <td>3.066348</td>
      <td>3.104617</td>
      <td>3.086429</td>
      <td>3.096426</td>
      <td>3.142041</td>
      <td>3.083848</td>
      <td>3.467144</td>
      <td>2.673882</td>
      <td>2.814478</td>
      <td>2.602801</td>
      <td>3.192560</td>
      <td>3.120171</td>
      <td>3.124507</td>
      <td>3.450264</td>
      <td>2.969699</td>
    </tr>
  </tbody>
</table>
</div>



### Is there a relationship between departure delay and passenger satisfaction?

#### Let's explore whether longer departure delays are associated with lower satisfaction ratings.


```python
# Calculate average satisfaction for different departure delay ranges
  
airline_data['Departure Delay Range in minutes'] = pd.cut(airline_data['Departure Delay'], 
                                         
                                         bins=[-np.inf, 10, 20, np.inf], labels=['<10', '10-20', '>20'])

average_satisfaction_by_delay = airline_data.groupby('Departure Delay Range in minutes')[['Satisfied']].mean()

average_satisfaction_by_delay.round(2)
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
      <th>Satisfied</th>
    </tr>
    <tr>
      <th>Departure Delay Range in minutes</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>&lt;10</th>
      <td>0.46</td>
    </tr>
    <tr>
      <th>10-20</th>
      <td>0.41</td>
    </tr>
    <tr>
      <th>&gt;20</th>
      <td>0.36</td>
    </tr>
  </tbody>
</table>
</div>



### Does the satisfaction with online services (booking, check-in, boarding) impact overall satisfaction?

#### We can analyze how satisfaction with online services affects the overall satisfaction level of passengers.


```python
import pandas as pd

# List of columns related to online services
online_services = [
    'Online Boarding',
    'Ease of Online Booking',
    'Check-in Service'
]

# Create a new DataFrame with selected columns
online_services_experience = airline_data[online_services]

# Calculate the mean overall satisfaction for each level of online service satisfaction
overall_satisfaction_mean = airline_data.groupby('Gender')[['Online Boarding', 'Ease of Online Booking', 'Check-in Service' ]].mean().reset_index()


# Calculate the overall satisfaction for each row (axis=1 means calculate across columns)

overall_satisfaction_mean['Overall Satisfaction'] = overall_satisfaction_mean.mean(axis=1)

overall_satisfaction_mean.round(2)
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
      <th>Gender</th>
      <th>Online Boarding</th>
      <th>Ease of Online Booking</th>
      <th>Check-in Service</th>
      <th>Overall Satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>3.31</td>
      <td>2.75</td>
      <td>3.30</td>
      <td>3.12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>3.19</td>
      <td>2.77</td>
      <td>3.32</td>
      <td>3.09</td>
    </tr>
  </tbody>
</table>
</div>

