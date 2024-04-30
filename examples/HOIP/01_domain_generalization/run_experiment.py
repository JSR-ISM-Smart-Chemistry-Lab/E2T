"""
"""
from pathlib import Path

import joblib
import pandas as pd
from lightning.pytorch.cli import LightningCLI
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from E2T.core import (
    LtGraphMNNs,
    CIFDataModule,
)
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
        pd.DataFrame([result]).to_csv(f"{out_dir}/result.csv", index=False)

    def _eval_result(self, support_size, out_dir):
        test = Batch.from_data_list(self.datamodule.test_dataset).to(self.model.device)

        if support_size > 0:
            if len(self.datamodule.train_val_dataset) < support_size:
                LOGGER.warning("Inference support size is larger than the dataset size")
                return dict()

            support = DataLoader(self.datamodule.train_val_dataset, batch_size=support_size, shuffle=True)
        else:
            support = Batch.from_data_list(self.datamodule.train_val_dataset).to(self.model.device)

        scaler = self.datamodule.scaler if self.datamodule.scale_y else None
        if scaler:
            support_y = scaler.inverse_transform(support.y.cpu().numpy().reshape(-1, 1))
            test_y = test.y

            support_pred = self.model.predict(support, support, scaler=scaler)
            test_pred = self.model.predict(support, test, scaler=scaler)

        else:
            support_y = support.y
            test_y = test.y

            support_pred = self.model.predict(support, support)
            test_pred = self.model.predict(support, test)

        out_dir_child = out_dir / f"support_{support_size}"
        result = evaluate_regression_result(support_y, support_pred, test_y, test_pred, out_dir=out_dir_child)
        result["support_size"] = support_size
        result["rand_id"] = self.datamodule.random_seed
        result["train_size"] = len(self.datamodule.train_dataset)
        result["val_size"] = len(self.datamodule.val_dataset) if self.datamodule.val_ratio > 0 else 0
        result["test_size"] = len(self.datamodule.test_dataset)
        result["out_dir"] = out_dir_child

        return result


def cli_main():
    cli = CustomLightningCLI(
        model_class=LtGraphMNNs,
        datamodule_class=CIFDataModule,
    )

if __name__ == '__main__':
    cli_main()
