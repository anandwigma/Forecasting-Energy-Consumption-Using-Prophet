# Forecasting-Energy-Consumption-Using-Prophet
### Brief Introduction
Prophet is a forecasting for time series data method developed and released by Facebook's data science team that open source for everyone.  What special about this method is it can plot the trend line of the data in yearly, monthly, weekly, and even daily. Additionally, it can add holiday and custom season effect to the forecast.


### The Data
In this tutorial, we use hourly electricity usage/consumption from Kentucky state in USA from June 2013 to August 2018. This data obtained from kaggle [here](https://www.kaggle.com/robikscube/hourly-energy-consumption#EKPC_hourly.csv). There are 45334 rows data that consist of timestamp and energy usage in MegaWatt. ![Preview of the data](https://raw.githubusercontent.com/anandwigma/Forecasting-Energy-Consumption-Using-Prophet/master/images/EKPC%20data.png)

## The Code
First load the data and named the column 'ds' for the timestamp/datetime and 'y' for the data in this case the electricity usage.
```python
import pandas as pd

import numpy as np
from fbprophet import Prophet

%matplotlib inline
import matplotlib.pyplot as plt

from IPython.display import display

sample=pd.read_csv('EKPC_hourly.csv')
```

After that make the model and fit it with the electricity usage data.
```python
from fbprophet.plot import add_changepoints_to_plot
m = Prophet(changepoint_prior_scale=0.01)
m.fit(sample)
```

Then make the forecast, specify the lenght and the type of time ('H' = hourly, 'Y' = yearly)
```python
future = m.make_future_dataframe(periods=8760, freq='H')
fcst = m.predict(future)
```

Last is plot the forecast graph
```python
fig = m.plot(fcst)
```

