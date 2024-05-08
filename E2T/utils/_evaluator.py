"""
Result evaluator
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_regression_result(
        y1_obs,
        y1_pred,
        y2_obs=None,
        y2_pred=None,
        target_name=None,
        out_dir=None,
        y1_label="train",
        y2_label="test",
    ):
    result = dict()

    # preprocess
    y1_obs = to_1darray(y1_obs)
    y1_pred = to_1darray(y1_pred)
    y2_obs = to_1darray(y2_obs)
    y2_pred = to_1darray(y2_pred)

    # Train
    result[f"R2_{y1_label}"] = r2_score(y1_obs, y1_pred) if len(y1_obs) > 1 else None
    result[f"corr2_{y1_label}"] = np.corrcoef(y1_obs, y1_pred)[0, 1]**2. if len(y1_obs) > 1 else None
    result[f"MAE_{y1_label}"] = mean_absolute_error(y1_obs, y1_pred) if len(y1_obs) > 1 else None
    result[f"MSE_{y1_label}"] = mean_squared_error(y1_obs, y1_pred) if len(y1_obs) > 1 else None
    result[f"RMSE_{y1_label}"] = mean_squared_error(y1_obs, y1_pred, squared=False) if len(y1_obs) > 1 else None

    # Test
    result[f"R2_{y2_label}"] = r2_score(y2_obs, y2_pred) if len(y2_obs) > 1 else None
    result[f"corr2_{y2_label}"] = np.corrcoef(y2_obs, y2_pred)[0, 1]**2. if len(y2_obs) > 1 else None
    result[f"MAE_{y2_label}"] = mean_absolute_error(y2_obs, y2_pred) if len(y2_obs) > 1 else None
    result[f"MSE_{y2_label}"] = mean_squared_error(y2_obs, y2_pred) if len(y2_obs) > 1 else None
    result[f"RMSE_{y2_label}"] = mean_squared_error(y2_obs, y2_pred, squared=False) if len(y2_obs) > 1 else None

    # Other info
    result[f"{y1_label}_size"] = len(y1_obs)
    result[f"{y2_label}_size"] = len(y2_obs)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # To escape matplotlib error
        import matplotlib
        matplotlib.use("Agg")
        if target_name is None:
            target_name = ""

        # output results
        result_df = pd.DataFrame([result])
        result_df.to_csv(out_dir / f"evaluation_result.csv", index=False)

        # output plot
        parity_plot(
            y1_obs,
            y1_pred,
            y2_obs,
            y2_pred,
            target_label=target_name,
            y1_label=y1_label,
            y2_label=y2_label,
            filename=out_dir / f"parity_plot_{y1_label}_{y2_label}.png",
        )

        return result


def to_1darray(array):
    if array is None:
        return np.array(list())
    if isinstance(array, torch.Tensor):
        array = array.cpu()

    return np.array(array).ravel()


def parity_plot(
        y1_obs,
        y1_pred,
        y2_obs,
        y2_pred,
        point_size=-1,
        y1_label="train",
        y2_label="test",
        target_label="",
        filename=None,
        labelsize=20,
        fontsize=14,
        palette="bright",
    ):
    OBS = "Observation"
    PRED = "Prediction"
    DATA = "data"

    y1_obs = to_1darray(y1_obs)
    y1_pred = to_1darray(y1_pred)
    y2_obs = to_1darray(y2_obs)
    y2_pred = to_1darray(y2_pred)

    # preprocess (dropna)
    df1 = pd.DataFrame({OBS: y1_obs, PRED: y1_pred}).dropna()
    df1[DATA] = y1_label

    df2 = pd.DataFrame({OBS: y2_obs, PRED: y2_pred}).dropna()
    df2[DATA] = y2_label

    df_all = pd.concat([df1, df2], axis=0).reset_index(drop=True)

    lim = [
        df_all[[OBS, PRED]].min().min(),
        df_all[[OBS, PRED]].max().max(),
    ]
    diff = lim[1] - lim[0]
    lim = [lim[0] - diff * 0.1, lim[1] + diff * 0.1]

    if point_size == -1:
        if len(y1_obs) + len(y2_obs) > 1000:
            point_size = 15
        elif len(y1_obs) + len(y2_obs) > 200:
            point_size = 25
        elif len(y1_obs) + len(y2_obs) > 100:
            point_size = 35
        else:
            point_size = 40

    # Plot
    g = sns.jointplot(
        data=df_all,
        x=OBS,
        y=PRED,
        hue=DATA,
        alpha=0.6,
        s=point_size,
        height=7,
        marginal_kws=dict(common_norm=False),
        palette=palette,
    )
    ax = g.ax_joint
    ax.plot(lim, lim, "--", lw=1, color="gray")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    if target_label:
        ax.set_xlabel(f"{OBS} {target_label}", fontsize=labelsize)
        ax.set_ylabel(f"{PRED} {target_label}", fontsize=labelsize)
    else:
        ax.set_xlabel(OBS, fontsize=labelsize)
        ax.set_ylabel(PRED, fontsize=labelsize)
    ax.set_aspect("equal")
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.legend(loc="lower right", fontsize=fontsize)

    ax.text(0.05, 0.98, f"[{y1_label} | {y2_label}]", transform=ax.transAxes, fontsize=fontsize)
    R2_train = r2_score(y1_obs, y1_pred) if len(y1_obs) > 1 else np.nan
    R2_test = r2_score(y2_obs, y2_pred) if len(y2_obs) > 1 else np.nan
    ax.text(0.05, 0.92, f"$R^2$: {R2_train:.3f} | {R2_test:.3f}", transform=ax.transAxes, fontsize=fontsize)

    rmse_train = mean_squared_error(y1_obs, y1_pred, squared=False) if len(y1_obs) > 1 else np.nan
    rmse_test = mean_squared_error(y2_obs, y2_pred, squared=False) if len(y2_obs) > 1 else np.nan
    ax.text(0.05, 0.875, f"RMSE: {rmse_train:.3g} | {rmse_test:.3g}", transform=ax.transAxes, fontsize=fontsize)

    mae_train = mean_absolute_error(y1_obs, y1_pred) if len(y1_obs) > 1 else np.nan
    mae_test = mean_absolute_error(y2_obs, y2_pred) if len(y2_obs) > 1 else np.nan
    ax.text(0.05, 0.83, f"MAE: {mae_train:.3g} | {mae_test:.3g}", transform=ax.transAxes, fontsize=fontsize)

    if filename:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, bbox_inches="tight")

    plt.close()

    return g

def save_loss_series_fig(loss_path, out_dir=None, xlabel="epoch", xfactor=1):
    """ Visualize series of loss log from CSVLogger of pytorch-lightning """
    if out_dir is None:
        out_dir = "."
    out_dir = Path(out_dir)

    df_loss = pd.read_csv(loss_path)
    df_loss[xlabel] = df_loss["epoch"].apply(lambda x: (x + 1) * xfactor)

    if "val_loss" in df_loss.columns:
        loss_cols = ["train_loss", "val_loss"]
    else:
        loss_cols = ["train_loss"]

    plt.figure()
    sns.lineplot(df_loss.groupby(xlabel)[loss_cols].mean(), alpha=0.3)
    plt.grid()
    plt.savefig(out_dir / "loss.png")
    plt.yscale("log")
    plt.savefig(out_dir / "loss_log.png")
    plt.close()
