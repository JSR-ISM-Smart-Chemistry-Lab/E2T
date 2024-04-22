# Domain Generalization Experiment

This guide will walk you through the process of running the E2T model with sample data. Please note that this example is designed for tabular data with a continuous value target. If you're working with a different type of data, you may need to make some modifications.

## Step 1: Configure the YAML File

All the arguments you'll need are listed in the YAML file. Here are the main ones you'll need to configure:

- `trainer.logger`: This argument is for the logger used by the trainer.
- `trainer.callbacks`: This argument is for the callbacks used by the trainer.
- `model.encoder`: This argument is for the encoder of MNNs.
- `model.header`: This argument is for the header of the model.
- `model.lr`: This argument is for the learning rate of the model.
- `data.csv_path`: This argument is for the path to the CSV file that contains the features (x), target (y), and class label.
- `data.train_classes`: Classes used in training.
- `data.sampling_policy`: The sampling policy to create episodes.
    - `GKFold`, `GroupKFold`: Different classes are used for support and query set.
    - `SoftGKFold`, `SoftGroupKFold`: Support and query set are sampled independently.
    - `Same`: Same sampled data are used for both support and query set.
- `data.support_nways`: the number of classes containing in each support.
- `data.support_size`:  The size of support. For example, support_nways = 5 and support_size = 20, then each class has 4 data.
- `data.query_nways`: the number of classes containing in each query.
- `data.query_size`: The size of query.
- `data.val_ratio`: The validation ratio to monitor learning curve.
- `data.target_col_idx`:  the index of the target column.
- `data.class_col_idx`: the index of the class column.

## Step 2: Run the Script

Once you've configured your YAML file, you can run the script with the following command:

```bash
$ python run_experiment.py fit --config [yamlfile]
```

