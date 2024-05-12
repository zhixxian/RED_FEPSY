import logging
from terminaltables import AsciiTable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# import wandb

from evaluator import DatasetEvaluator
from metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss, OELoss, UELoss, MCELoss
from torch_helper import to_numpy

logger = logging.getLogger(__name__)


class CalibrateEvaluator(DatasetEvaluator):
    def __init__(self, num_classes, num_bins=15, device="cuda:0") -> None:
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.logits = None
        self.labels = None

    def num_samples(self):
        return (
            self.labels.shape[0]
            if self.labels is not None
            else 0
        )

    def main_metric(self) -> None:
        return "ece"

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """update

        Args:
            logits (torch.Tensor): n x num_classes
            label (torch.Tensor): n x 1
        """
        assert logits.shape[0] == labels.shape[0]
        if self.logits is None:
            self.logits = logits
            self.labels = labels
        else:
            self.logits = torch.cat((self.logits, logits), dim=0)
            self.labels = torch.cat((self.labels, labels), dim=0)

    def mean_score(self, print=False, all_metric=True, print_classes=True):
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = ECELoss(self.num_bins).to(self.device)
        mce_criterion = MCELoss(self.num_bins).to(self.device)
        aece_criterion = AdaptiveECELoss(self.num_bins).to(self.device)
        cece_criterion = ClasswiseECELoss(self.num_bins).to(self.device)
        oe_criterion = OELoss(self.num_bins).to(self.device)
        ue_criterion = UELoss(self.num_bins).to(self.device)

        nll = nll_criterion(self.logits, self.labels).item()
        ece = ece_criterion(self.logits, self.labels).item()
        mce= mce_criterion(self.logits, self.labels).item()
        aece = aece_criterion(self.logits, self.labels).item()
        per_class_ece = cece_criterion(self.logits, self.labels, print_classes=print_classes)
        # type(per_class_ece)
    
        oe = oe_criterion(self.logits, self.labels).item()
        ue = ue_criterion(self.logits, self.labels).item()

        # metric = {"nll": nll, "ece": ece, "aece": aece, "cece": cece}
        # metric = {"nll": nll, "ece": ece, "aece": aece, "cece": cece, "oe": oe, "ue": ue}
        metric = {"nll": nll, "ece": ece,"mce":mce, "aece": aece, "per_class_ece": per_class_ece, "oe": oe, "ue": ue}

        # columns = ["samples", "nll", "ece", "aece", "cece"]
        # columns = ["samples", "nll", "ece", "aece", "cece", "oe", "ue"]
        columns = ["samples", "nll", "ece",'mce', "aece", "per_class_ece", "oe", "ue"]
        table_data = [columns]
        # # table_data.append(
        # #     [
        # #         self.num_samples(),
        # #         "{:.5f}".format(nll),
        # #         "{:.5f}".format(ece),
        # #         "{:.5f}".format(aece),
        # #         "{:.5f}".format(cece),
        # #     ]
        # # )
        # table_data.append(
        #     [
        #         self.num_samples(),
        #         "{:.5f}".format(nll),
        #         "{:.5f}".format(ece),
        #         "{:.5f}".format(mce),
        #         "{:.5f}".format(aece),
        #         "{:.5f}".format(per_class_ece),
        #         "{:.5f}".format(oe),
        #         "{:.5f}".format(ue),
        #     ]
        # )

        if print:
            table = AsciiTable(table_data)
            logger.info("\n" + table.table)

        if all_metric:
            return metric, table_data
        else:
            return metric[self.main_metric()]

    def wandb_score_table(self):
        _, table_data = self.mean_score(print=False)
        return wandb.Table(
            columns=table_data[0],
            data=table_data[1:]
        )

    def save_npz(self, save_path):
        np.savez(
            save_path,
            logits=to_numpy(self.logits),
            labels=to_numpy(self.labels)
        )

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
