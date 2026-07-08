## Model Performance

**Note:** An earlier version of this analysis used a random train/test split, 
which produced an inflated R² of 0.59 due to temporal data leakage (train and 
test rows were only minutes apart in this correlated time-series dataset).

After correcting to a chronological split (train on earlier months, test on 
later months — simulating real deployment), the honest results are:

| Model | R² | Notes |
|---|---|---|
| Naive baseline (24h rolling average) | 0.019 | Performance floor |
| Random Forest (regularized) | -0.658 | Fails to extrapolate to unseen future conditions |
| **Linear Regression** | **0.120** | Best honest result; selected for deployment |

**Why Linear Regression over Random Forest:** tree-based models cannot 
extrapolate beyond the value ranges seen during training. Once evaluated on a 
genuinely future time period, Random Forest and other tree ensembles performed 
worse than a naive guess. Linear Regression, while modest in absolute 
performance, generalizes more reliably.

**Takeaway:** ambient sensor data (temperature, humidity, time) has a 
relatively low ceiling for predicting appliance-level energy usage — capturing 
more of the real driver (occupant behavior) would likely require occupancy 
sensors or appliance-level usage logs.
