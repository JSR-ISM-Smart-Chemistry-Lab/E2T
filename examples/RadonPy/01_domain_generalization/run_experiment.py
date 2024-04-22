"""

"""

from pathlib import Path
import joblib
import pandas as pd
import torch
from lightning.pytorch.cli import LightningCLI

from E2T.core import TableDataModule, LtMNNs
from E2T.utils import (
    evaluate_regression_result,
    save_loss_series_fig,
    set_directory,
    LOGGER,
)


class CustomLightningCLI(LightningCLI):
    def before_instantiate_classes(self):
        try:
            self.save_config_kwargs["config_filename"] = f"{self.config.fit.trainer.logger.init_agrs.name}.yaml"
        except:
            LOGGER.info("Output config.yaml")
        else:
            LOGGER.info(f"Output {self.save_config_kwargs['config_filename']}")

    def after_fit(self):
        LOGGER.info("Finished training")
        if self.datamodule.scale_y:
            joblib.dump(self.datamodule.scaler, f"{self.trainer.logger.log_dir}/scaler.joblib")

        out_dir = set_directory(f"{self.trainer.logger.log_dir}/eval", prefix=False)
        self.eval_result(out_dir)

        # for csv logger
        loss_path = f"{self.trainer.logger.log_dir}/metrics.csv"
        if Path(loss_path).exists():
            save_loss_series_fig(loss_path, out_dir, xlabel="Episodes", xfactor=self.datamodule.train_epoch_length)

    def eval_result(self, out_dir):
        result = self._eval_result(support_size=-1, out_dir=out_dir)
        pd.DataFrame([result]).to_csv(out_dir / "eval_result.csv", index=False)

    def _eval_result(self, support_size, out_dir):
        template_X = self.datamodule.train_dataset.X
        template_y = self.datamodule.train_dataset.y

        train_val_data_sampled = self.datamodule.train_val_data_sampled
        test_data = self.datamodule.test_data

        support = train_val_data_sampled
        if support_size > 0:
            if len(support) < support_size:
                return dict()
            support = support.sample(support_size, random_state=self.datamodule.random_seed)

        target_col, class_col = self.datamodule.target_col, self.datamodule.class_col
        support_x, support_y = (
            torch.tensor(support.drop(columns=[target_col, class_col]).values).to(template_X),
            torch.tensor(support[target_col].values).to(template_y).to(template_y),
        )
        test_x, test_y = (
            torch.tensor(test_data.drop(columns=[target_col, class_col]).values).to(template_X),
            torch.tensor(test_data[target_col].values).to(template_y).to(template_y),
        )

        scaler = self.datamodule.scaler if self.datamodule.scale_y else None
        if scaler:
            support_y_scaled = torch.tensor(scaler.transform(support_y.cpu().reshape(-1, 1)).reshape(-1)).to(template_y)
            support_y_pred = self.model.predict(support_x, support_y_scaled, support_x, scaler=scaler)
            test_y_pred = self.model.predict(support_x, support_y_scaled, test_x, scaler=scaler)
        else:
            support_y_pred = self.model.predict(support_x, support_y, support_x)
            test_y_pred = self.model.predict(support_x, support_y, test_x)

        out_dir_child = out_dir / f"support_{support_size}"
        result = evaluate_regression_result(support_y, support_y_pred, test_y, test_y_pred, out_dir=out_dir_child)
        result["support_size"] = support_size
        result["random_seed"] = self.datamodule.random_seed
        result["train_size"] = len(self.datamodule.train_data_sampled)
        result["val_size"] = len(self.datamodule.val_data_sampled) if self.datamodule.val_ratio > 0 else 0
        result["train_val_size"] = result["train_size"] + result["val_size"]
        result["test_size"] = len(test_data)
        result["out_dir"] = out_dir_child.absolute()
        result["csv_path"] = Path(self.datamodule.csv_path).absolute()
        result["samples_per_class"] = self.datamodule.samples_per_class
        result["train_classes"] = self.datamodule.train_classes

        return result

def cli_main():
    cli = CustomLightningCLI(
        model_class=LtMNNs,
        datamodule_class=TableDataModule,
    )

if __name__ == "__main__":
    cli_main()
