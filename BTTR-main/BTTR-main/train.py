from pytorch_lightning.utilities.cli import LightningCLI

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR
import torch
# print(torch.version.cuda)

# print(torch.cuda.device_count())

if __name__ ==  '__main__':
    cli = LightningCLI(LitBTTR, CROHMEDatamodule)
