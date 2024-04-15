# Domain generailization experiment

Follow these steps to run E2T with sample data.

This example supports tabular data with continuous value target. You need some modification to apply another type of data.

## Step1. Configure yaml file
All arguments are listed in the yaml file.
Here is the main arguments to configure:
- trainer.logger
- trainer.callbacks
- model.encoder
    - encoder of MNNs
- model.header
- model.lr
- data.csv_path
    - the csv file consist of features (x), target (y), and class label. 
- data.train_classes
- data.sampling_policy
- data.support_nways
- data.support_size
- data.query_nways
- data.query_size
- data.val_ratio
- data.target_col_idx
    - target column idx
- data.class_col_idx
    - class column idx

## Step2. Run script
```
$ python run_experiment.py fit --config [yamlfile]
```
