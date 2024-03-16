# AERO,  From Chaos to Clarity: Time Series Anomaly Detection in Astronomical Observations(ICDE 2024)

This is PyTorch implementation of AERO in the following paper: 

"From Chaos to Clarity: Time Series Anomaly Detection in Astronomical Observations"

## Requirements

Dependency can be installed using the following command:

```
pip install -r requirements.txt
```
## Data Preparation
```
# put your dataset under processed/ directory with the same structure shown in the data/msl/

Dataset_txt
 |-AstrosetMiddle
 | |-AstrosetMiddle_train.txt    # training data
 | |-AstrosetMiddle_test.txt     # test data
 | |-AstrosetMiddle_interpretation_label.txt    # True anomaly label
 |-your_dataset
 | |-XX_train.txt
 | |-XX_test.txt
 | |-XX_interpretation_label.txt
 | ...

```
### Notices:
* The row in XX_train.txt(XX_test.txt) represents a timestamp and the coloum represents a object. However, the first coloum represents timestamps.
* In interpretation_label.txt, every row represents a true anomaly segment. For example, "2200-2900:48" represents object 48 occurs a anomaly during 2200-2900 timestamps.
* The object number in XX_interpretation_label.txt starts from 1 instead of 0. 

## Dataset Preprocessing
Preprocess all datasets using the command
```
python3 src/processing.py AstrosetMiddle
```

## Model Training
- SyntheticMiddle
```
python3 main.py  --dataset_name SyntheticMiddle  --retrain --freeze_patience 5 --freeze_delta 0.01 --stop_patience 5 --stop_delta 0.01
```

- SyntheticHigh
```
python3 main.py  --dataset_name SyntheticHigh  --retrain --freeze_patience 5 --freeze_delta 0.01 --stop_patience 5 --stop_delta 0.005
```

- SyntheticLow
```
python3 main.py  --dataset_name SyntheticLow  --retrain --freeze_patience 5 --freeze_delta 0.01 --stop_patience 5 --stop_delta 0.005
```

- AstrosetMiddle
```
python3 main.py  --dataset_name AstrosetMiddle  --retrain --freeze_patience 5 --freeze_delta 0.01 --stop_patience 5 --stop_delta 0.005
```

- AstrosetHigh
```
python3 main.py  --dataset_name AstrosetHigh  --retrain --freeze_patience 5 --freeze_delta 0.01 --stop_patience 5 --stop_delta 0.005
```

- AstrosetLow
```
python3 main.py  --dataset_name AstrosetLow  --retrain --freeze_patience 5 --freeze_delta 0.005 --stop_patience 5 --stop_delta 0.001
```


## Run the trained Model

You can run the following command to evaluate the test datasets using the trained model.

```
python3 main.py  --dataset_name XX --test
```








