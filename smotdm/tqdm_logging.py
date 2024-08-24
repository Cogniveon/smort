import logging
import sys

import pytorch_lightning as pl
import tqdm
from pytorch_lightning.callbacks import \
    TQDMProgressBar as OriginalTQDMProgressBar


# from https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def customize_bar(bar):
    if not sys.stdout.isatty():
        bar.disable = True
    bar.leave = True  # remove the bar after completion
    return bar


class TQDMProgressBar(OriginalTQDMProgressBar):
    # remove the annoying v_num in the bar
    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        items_dict = super().get_metrics(trainer, pl_module).copy()

        if "v_num" in items_dict:
            items_dict.pop("v_num")
        return items_dict

    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
        return customize_bar(bar)

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        return customize_bar(bar)

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        return customize_bar(bar)

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        return customize_bar(bar)
