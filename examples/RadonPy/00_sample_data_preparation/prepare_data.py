
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from scipy.sparse import coo_matrix


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SMILES = "smiles"
POLYMER_CLASS = "polymer_class"
CP = "Cp"
REQUIRED_COLUMNS = [SMILES, POLYMER_CLASS, CP]

def inspect_data(data):
    # Required columns
    for col in REQUIRED_COLUMNS:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")


def calculate_ecfp(data):
    """
    Calculate ECFP count type descriptor for sample data
    """
    smiles_list = data[SMILES].tolist()

    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

    ecfp_matrix = np.vstack([mol2fp(mol) for mol in mols])

    df_ecfp = pd.DataFrame(
        ecfp_matrix,
        columns=[f"ecfpct_{i}" for i in range(ecfp_matrix.shape[1])],
        index=data.index,
    )

    return df_ecfp


def mol2fp(mol, radius=3, nBits=2048):
    """
    Convert a molecule to ECFP count type fingerprint
    """
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits)
    fp_dict = fp.GetNonzeroElements()
    fp_array = dict2fp(fp_dict, nBits)

    return fp_array


def dict2fp(dictionary, nBits=2048):
    rows = np.array(list(dictionary.keys()))
    cols = np.zeros_like(rows)
    data = np.array(list(dictionary.values()))

    output_matrix = coo_matrix((data, (rows, cols)), shape=(nBits, 1))
    output_list = output_matrix.toarray().flatten()

    return output_list.tolist()

if __name__ == "__main__":
    # parse data path
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, help="Path to the sample data installed from RadonPy package", default="PI1070.csv")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save the preprocessed data", default="PI1070_preprocessed.csv")
    args = parser.parse_args()

    LOGGER.info("Load data from %s", args.data_path)
    data = pd.read_csv(args.data_path)

    LOGGER.info("Inspect data")
    inspect_data(data)

    LOGGER.info("Calculate descriptor")
    descriptors = calculate_ecfp(data)

    # Combine data
    descriptors[POLYMER_CLASS] = data[POLYMER_CLASS]
    descriptors[CP] = data[CP]

    descriptors = descriptors.dropna()

    # Save preprocessed data
    LOGGER.info("Output data to %s", args.output_path)
    p = Path(args.output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    descriptors.to_csv(args.output_path, index=False)

    LOGGER.info("Done")
