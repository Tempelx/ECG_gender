# ECG Gender Estimation

Classification project for estimating the gender in normal 12-lead ECG signals.

### Running

For the feature calculation the Chinese Physiological Signal Challenge 2018 Database is used:
```
http://2018.icbeb.org/Challenge.html
```
If you want to recalculate the features go to this website and store the data on your local disc.
Afterwards uncomment and change the path accordingly in main.py.
```
path_to_chin = 'your_path_goes_here'
features = fe.feature_calc(path_to_chin)
```
Otherwise just run the training function.
