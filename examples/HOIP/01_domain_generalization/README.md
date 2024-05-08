# Domain generalization in HOIP dataset

This example is based on the ANE package and the HOIP dataset, so you will need to make additional modifications to use it in a differenct application.

## Step 1: Configure the YAML File

All the arguments you'll need are listed in the YAML file. Here are the main ones you'll need to configure:

- `trainer.logger`: This argument is for the logger used by the trainer.
- `trainer.callbacks`: This argument is for the callbacks used by the trainer.
- `model.gnn_model`: This argument is for the encoder GNN model of MNNs. The default value is mpnn (message passing neural network)
- `model.header`: This argument is for the header of the model.
- `model.lr`: This argument is for the learning rate of the model.
- `data.train_path`: This argument is for the path to the training data directory.
- `data.train_id_target_file`: Training metadata filename.
- `data.test_path`: This argument is for the path to the test data directory.
- `data.test_id_target_file`: Test metadata filename.
- `data.sampling_policy`: The sampling policy to create episodes.
    - `GKFold`, `GroupKFold`: Different classes are used for support and query set.
    - `SoftGKFold`, `SoftGroupKFold`: Support and query set are sampled independently.
    - `Same`: Same sampled data are used for both support and query set.
- `data.support_nways`: the number of classes containing in each support.
- `data.support_size`:  The size of support. For example, support_nways = 5 and support_size = 20, then each class has 4 data.
- `data.query_nways`: the number of classes containing in each query.
- `data.query_size`: The size of query.
- `data.val_ratio`: The validation ratio to monitor learning curve.

## Step 2: Run the Script

Once you've configured your YAML file, you can run the script with the following command:

```bash
$ python run_experiment.py fit --config [yamlfile]
```

