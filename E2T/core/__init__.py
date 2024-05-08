
from ._datamodule import (
    TableDataModule,
    TableDataset,
    CustomTaskDataset,
)

from ._module import (
    FCEncoder,
    LtMNNs,
    RidgeRegressionHeader,
)

# Since graph module requires some additional packages,
# importing graph module is not mandatory.
try:
    from ._graph_datamodule import (
        CIFDataModule,
        CIFDataset,
        CIFCustomTaskDataset,
        pyg_collate,
    )

    from ._graph_module import (
        LtGraphMNNs,
    )
except ImportError:
    pass


__all__ = [
    "TableDataModule",
    "TableDataset",
    "CustomTaskDataset",
    "FCEncoder",
    "LtMNNs",
    "RidgeRegressionHeader",
    "CIFDataModule",
    "CIFDataset",
    "CIFCustomTaskDataset",
    "pyg_collate",
    "LtGraphMNNs",
]
