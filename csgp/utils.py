import sys
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
import gpytorch
import numpy as np


class ScaleToBounds(torch.nn.Module):
    def __init__(self, lower_bound, upper_bound):
        super().__init__()
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.register_buffer("min_val", torch.tensor(lower_bound))
        self.register_buffer("max_val", torch.tensor(upper_bound))

    def forward(self, x):
        if self.training:
            min_val = x.min()
            max_val = x.max()
            self.min_val.data = min_val
            self.max_val.data = max_val
        else:
            min_val = self.min_val
            max_val = self.max_val
            # Clamp extreme values
            x = x.clamp(min_val, max_val)

        diff = max_val - min_val
        x = (x - min_val) * (0.95 * (self.upper_bound - self.lower_bound) / diff) + 0.95 * self.lower_bound
        return x
    

def ece_score(py, y_test, n_bins=15):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if a < py_value[i] <= b:
                bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if bm[m] != 0:
            acc[m] = acc[m] / bm[m]
            conf[m] = conf[m] / bm[m]
    ece = 0
    for m in range(n_bins):
        ece += bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(bm) * 100.


def expected_calibration_error(y_true, y_prob, n_bins=15, return_bin_info=False):
    y_true, y_prob = y_true.cpu(), y_prob.cpu()
    if y_prob.ndim == 1:
        confs, accs = y_prob, y_true
    elif y_prob.ndim == 2:
        confs, y_pred = y_prob.max(dim=1)
        accs = y_true.eq(y_pred).float()
    else:
        raise ValueError(f'y_prob shape `{y_prob.shape}` is not valid')

    bin_bounds = torch.linspace(0, 1, n_bins + 1)
    bin_indices = torch.bucketize(confs, bin_bounds[1:-1])
    bin_counts = torch.bincount(bin_indices, minlength=n_bins)
    bin_confs = torch.bincount(bin_indices, weights=confs, minlength=n_bins) / bin_counts
    bin_accs = torch.bincount(bin_indices, weights=accs, minlength=n_bins) / bin_counts
    bin_confs[bin_confs.isnan()] = 0
    bin_accs[bin_accs.isnan()] = 0
    ece = ((bin_accs - bin_confs).abs().double() @ bin_counts.double()).item() / len(y_true)

    if return_bin_info:
        return ece, bin_accs, bin_confs, bin_counts

    return ece * 100.


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

class RegressionMetrics:
    def __init__(self, pred_mean, pred_var, test_y) -> None:
        self.pred_mean = pred_mean.reshape(-1)  # [n_test]-size tensor
        self.pred_var = pred_var.reshape(-1).clamp_min(1e-8)  # [n_test]-size tensor
        self.test_y = test_y.reshape(-1)  # [n_test]-size tensor
        self.metrics = {}

    def rmse(self):
        pred_mean = self.pred_mean  # [n_test]-size tensor
        test_y = self.test_y  # [n_test]-size tensor

        rmse_val = ((test_y - pred_mean).detach() ** 2).mean().sqrt().item()  # a number
        self.metrics['rmse'] = rmse_val
        return rmse_val

    def nll(self):
        pred_mean = self.pred_mean  # [n_test]-size tensor
        pred_var = self.pred_var  # [n_test]-size tensor
        test_y = self.test_y  # [n_test]-size tensor

        nll_val = gpytorch.metrics.gaussian_nll(pred_mean, pred_var.diag(), test_y).item()  # a number
        self.metrics['nll'] = nll_val

        return nll_val

    def nlpd(self):
        # pred_mean = self.pred_mean  # [n_test]-size tensor
        # pred_var = self.pred_var  # [n_test]-size tensor
        # test_y = self.test_y  # [n_test]-size tensor
        #
        # model_eval_at_test_x = gpytorch.distributions.MultivariateNormal(pred_mean, pred_var.diag(0))
        # likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.test_y.device)
        # trained_pred_dist = likelihood(model_eval_at_test_x)
        #
        # nlpd_val = gpytorch.metrics.negative_log_predictive_density(trained_pred_dist, test_y).item()  # a number
        # self.metrics['nlpd'] = nlpd_val

        pred_mean = self.pred_mean  # [n_test]-size tensor
        pred_var = self.pred_var  # [n_test]-size tensor
        test_y = self.test_y  # [n_test]-size tensor

        nlpd_val = 0.5 * torch.log(2 * torch.pi * pred_var) + 0.5 * ((test_y - pred_mean) ** 2) / pred_var
        nlpd_val = nlpd_val.mean().item()  # a number
        self.metrics['nlpd'] = nlpd_val

        return nlpd_val

    def coverage_score(self, num_std=1):
        pred_mean = self.pred_mean  # [n_test]-size tensor
        pred_var = self.pred_var  # [n_test]-size tensor
        test_y = self.test_y  # [n_test]-size tensor

        pred_lower = pred_mean - num_std * pred_var.sqrt()
        pred_upper = pred_mean + num_std * pred_var.sqrt()
        coverage_bool = (test_y >= pred_lower) & (test_y <= pred_upper)

        coverage_score_val = (coverage_bool.sum() / len(coverage_bool)).item()

        self.metrics['coverage_score'] = coverage_score_val

        return coverage_score_val


class ClassificationMetrics:
    def __init__(self, num_mc=20, n_bins=15, option='logits'):
        self.num_mc = num_mc
        self.n_bins = n_bins
        self.option = option
        assert self.option in ['logits', 'probs']

    def accuracy(self, model, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(model.device), target.to(model.device)
                output_ = []
                for mc_run in range(self.num_mc):
                    output, _ = model(data)
                    output_.append(output)
                output = torch.mean(torch.stack(output_), dim=0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        acc = 100. * correct / total
        return acc

    def nll(self, model, test_loader):
        nll = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(model.device), target.to(model.device)
                output_ = []
                for mc_run in range(self.num_mc):
                    output, _ = model(data)
                    output_.append(output)
                output = torch.mean(torch.stack(output_), dim=0)
                if self.option == 'logits':
                    nll += F.cross_entropy(output, target, reduction='sum').item()
                elif self.option == 'probs':
                    nll += F.nll_loss(torch.log(output), target, reduction='sum').item()
        nll /= len(test_loader.dataset)
        return nll

    def ece(self, model, test_loader):
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(model.device), target.to(model.device)
                output_ = []
                for mc_run in range(self.num_mc):
                    output, _ = model(data)
                    output_.append(output)
                output = torch.mean(torch.stack(output_), dim=0)
                if self.option == 'logits':
                    all_probs.append(F.softmax(output, dim=1))
                elif self.option == 'probs':
                    all_probs.append(output)
                all_labels.append(target)

        probs = torch.cat(all_probs)
        labels = torch.cat(all_labels)

        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=probs.device)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    

class MinMaxNormalize:
    """
    Min-max normalization to [0,1]
    """

    def __call__(self, sample):
        return (sample - sample.min()) / (sample.max() - sample.min())


class PrintOutput:
    """
    # A class to print the output to both terminal and file
    """

    def __init__(self, file):
        self.file = file
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)  # print to the terminal
        self.file.write(message)  # write to the file

    def flush(self):
        self.terminal.flush()
        self.file.flush()
