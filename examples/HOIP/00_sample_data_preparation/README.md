# Preparation for HOIP experiment

Follow these steps to prepare sample data:

## Step 1. Clone ANE repository
First, clone ANE repository (https://github.com/ngs00/ane).
We can find preprocessed HOIP dataset in the materials_property_prediction/datasets directory. Here, hoip_high and hoip_low are HOIP_GeF and HOIP_PbI in our notation, respectively.

## Step 2. Install ane package locally
To use functions in ANE, we install ANE locally. Run following command after adding setup.py to the ANE repository.

```
pip install -e {your path to the ANE repository}
```

Here is a simple setup.py example for ANE.
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="ane",
    version="1.0.0",
    packages=find_packages(),
)

```

## Step 3. Install other required packages
To run our following experiments, install the other required packages.
- torch_geometric
- mendeleev
- pymatgen
- openpyxl
