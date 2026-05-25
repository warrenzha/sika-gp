import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from csgp.layers.gps import CSGP
import time
import random
import os
import csv

BACKBONE = "distilbert-base-uncased"
OOD_LABEL = 150          # 你的数据里 max=150 且 unique_count=151 => intent==150 为 OOD/OOS
NUM_ID_LABELS = 150      # ID intents: 0..149


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.prior_sigma = prior_sigma
        self.w_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.w_rho = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.full((out_features,), -5.0))

    def _sigma(self, rho):
        return F.softplus(rho) + 1e-8

    def forward(self, x, return_kl=False, sparse=False):
        # Sample per-batch weights/biases so each sample uses its own draw
        w_sigma = self._sigma(self.w_rho)
        b_sigma = self._sigma(self.b_rho)

        batch_size = x.shape[0]
        w_eps = torch.randn((batch_size, *self.w_mu.shape), device=x.device, dtype=x.dtype)
        b_eps = torch.randn((batch_size, *self.b_mu.shape), device=x.device, dtype=x.dtype)
        w = self.w_mu + w_sigma * w_eps
        b = self.b_mu + b_sigma * b_eps

        logits = (w * x.unsqueeze(1)).sum(-1) + b
        kl = self.kl_div(w_sigma, b_sigma)
        if return_kl:
            return logits, kl
        return logits

    def kl_div(self, w_sigma, b_sigma):
        prior_var = self.prior_sigma ** 2

        def kl_gauss(mu, sigma):
            var = sigma ** 2
            return 0.5 * torch.sum((var + mu**2) / prior_var - 1.0 + torch.log(prior_var / var))

        return kl_gauss(self.w_mu, w_sigma) + kl_gauss(self.b_mu, b_sigma)


class KISSGPLayer(gpytorch.models.ApproximateGP):
    """KISSGP layer using grid interpolation variational strategy."""

    def __init__(self, num_dim, grid_bounds=(-10.0, 10.0), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self,
                grid_size=grid_size,
                grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ),
            num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DistilBERT_SVDKL(nn.Module):
    """DistilBERT backbone followed by a KISSGP head with Softmax likelihood."""

    def __init__(self, num_labels, proj_dim=256, grid_bounds=(-10.0, 10.0), grid_size=64):
        super().__init__()
        self.is_svdkl = True
        self.backbone = AutoModel.from_pretrained(BACKBONE)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, proj_dim)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bounds[0], grid_bounds[1])
        self.gp_layer = KISSGPLayer(num_dim=proj_dim, grid_bounds=grid_bounds, grid_size=grid_size)
        self.likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=proj_dim, num_classes=num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        z = out.last_hidden_state[:, 0, :]
        z = self.proj(z)
        z = self.scale_to_bounds(z)
        z = z.transpose(-1, -2).unsqueeze(-1)
        return self.gp_layer(z)

    def predict_proba(self, batch, num_samples=20):
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_samples):
            preds = self.likelihood(self(batch["input_ids"], batch["attention_mask"]))
            probs_T = preds.probs  # (S,B,C)
            mean_probs = probs_T.mean(dim=0)
        return probs_T, mean_probs


class DistilBERT_BNN(nn.Module):
    def __init__(self, num_labels, proj_dim=256, prior_sigma=1.0, dyadic_level=7, bayes="csgp"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(BACKBONE)
        hidden = self.backbone.config.hidden_size  # 768
        self.proj = nn.Linear(hidden, proj_dim)
        # Select Bayesian head implementation
        if bayes == "bayes_linear":
            self.bayes = BayesianLinear(proj_dim, num_labels, prior_sigma=prior_sigma)
        elif bayes == "csgp":
            self.bayes = CSGP(
                in_features=proj_dim,
                out_features=num_labels,
                dyadic_level=dyadic_level,
                anchor=False,
            )
        else:
            raise ValueError(f"Unsupported bayes head: {bayes}")

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        z = out.last_hidden_state[:, 0, :]  # CLS pooling
        z = self.proj(z)
        logits, kl = self.bayes(z, return_kl=True, sparse=True)
        return logits, kl


def set_backbone_trainable(model: DistilBERT_BNN, trainable: bool, unfreeze_last_n_layers: int = 2):
    # DistilBERT encoder layers are model.backbone.transformer.layer (len=6)
    for p in model.backbone.parameters():
        p.requires_grad = False
    if trainable:
        layers = model.backbone.transformer.layer
        for layer in layers[-unfreeze_last_n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True


def beta_warmup(step, warmup_steps):
    return 1.0 if warmup_steps <= 0 else min(1.0, step / warmup_steps)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def mc_probs(model, batch, T=20):
    """
    Returns:
      probs_T: (T,B,C)
      mean_probs: (B,C)
    """
    model.eval()
    if getattr(model, "is_svdkl", False):
        probs_T, mean_probs = model.predict_proba(batch, num_samples=T)
        return probs_T, mean_probs
    probs = []
    for _ in range(T):
        logits, _ = model(**batch)
        probs.append(torch.softmax(logits, dim=-1))
    probs_T = torch.stack(probs, dim=0)
    mean_probs = probs_T.mean(dim=0)
    return probs_T, mean_probs


@torch.no_grad()
def uncertainty_scores(probs_T, mean_probs):
    """
    entropy: predictive entropy H(E[p])
    mi: mutual information H(E[p]) - E[H(p)]
    """
    entropy = -(mean_probs * mean_probs.clamp_min(1e-9).log()).sum(dim=-1)  # (B,)
    ent_each = -(probs_T * probs_T.clamp_min(1e-9).log()).sum(dim=-1)       # (T,B)
    mi = entropy - ent_each.mean(dim=0)                                     # (B,)
    return entropy.cpu().numpy(), mi.cpu().numpy()


@torch.no_grad()
def compute_id_accuracy(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if getattr(model, "is_svdkl", False):
            _, mean_probs = model.predict_proba(batch, num_samples=20)
            pred = mean_probs.argmax(dim=-1)
        else:
            logits, _ = model(batch["input_ids"], batch["attention_mask"])
            pred = logits.argmax(dim=-1)
        y_true.extend(batch["labels"].cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
    return accuracy_score(y_true, y_pred)

@torch.no_grad()
def compute_id_nll(model, loader, device):
    model.eval()
    total_nll = 0.0
    total_count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if getattr(model, "is_svdkl", False):
            _, mean_probs = model.predict_proba(batch, num_samples=50)
            nll = F.nll_loss(mean_probs.log(), batch["labels"], reduction="sum")
        else:
            logits, _ = model(batch["input_ids"], batch["attention_mask"])
            nll = F.cross_entropy(logits, batch["labels"], reduction='sum')
        total_nll += nll.item()
        total_count += batch["labels"].size(0)
    return total_nll / total_count if total_count > 0 else float('inf')

@torch.no_grad()
def compute_id_ece(model, loader, device, n_bins=15):
    model.eval()
    confidences = []
    predictions = []
    labels = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if getattr(model, "is_svdkl", False):
            _, mean_probs = model.predict_proba(batch, num_samples=50)
            probs = mean_probs
        else:
            logits, _ = model(batch["input_ids"], batch["attention_mask"])
            probs = torch.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
        confidences.extend(conf.cpu().numpy().tolist())
        predictions.extend(pred.cpu().numpy().tolist())
        labels.extend(batch["labels"].cpu().numpy().tolist())

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = [(conf >= bin_lower) and (conf < bin_upper) for conf in confidences]
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean([pred == label for pred, label, inc in zip(predictions, labels, in_bin) if inc])
            avg_confidence_in_bin = np.mean([conf for conf, inc in zip(confidences, in_bin) if inc])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=2e-5)
    ap.add_argument("--kl_warmup_frac", type=float, default=0.15)
    ap.add_argument("--freeze_epochs", type=int, default=2)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_csv", type=str, default=None, help="Optional path to append metrics as CSV")
    ap.add_argument("--bayes", type=str, default="csgp", choices=["csgp", "bayes_linear", "svdkl"], help="Bayesian head type")
    ap.add_argument("--dyadic_level", type=int, default=7, help="Dyadic level for CSGP head when bayes=csgp")
    args = ap.parse_args()

    set_seed(args.seed)

    ds = load_dataset("clinc_oos", "plus")
    print("splits:", list(ds.keys()))
    print("train columns:", ds["train"].column_names)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_len)

    ds = ds.map(tok, batched=True)

    # ----- Split ID vs OOD using intent==150 -----
    def is_ood(x):
        return x["intent"] == OOD_LABEL

    def is_id(x):
        return x["intent"] != OOD_LABEL

    train_id = ds["train"].filter(is_id)  # train 通常不含 OOD；即便含，也过滤掉
    val_id   = ds["validation"].filter(is_id)
    test_id  = ds["test"].filter(is_id)

    val_ood  = ds["validation"].filter(is_ood)
    test_ood = ds["test"].filter(is_ood)

    print("val  ID/OOD:", len(val_id), len(val_ood))
    print("test ID/OOD:", len(test_id), len(test_ood))

    # ----- Use only 150 ID labels (0..149) for training -----
    # intent 已经是连续的 0..150，这里只需把列名改成 labels
    train_id = train_id.rename_column("intent", "labels")
    val_id   = val_id.rename_column("intent", "labels")
    test_id  = test_id.rename_column("intent", "labels")

    # OOD split labels 不用于 CE，但为了 loader 统一也 rename
    val_ood  = val_ood.rename_column("intent", "labels")
    test_ood = test_ood.rename_column("intent", "labels")

    # Sanity check: ensure no OOD label in ID splits
    for name, split in [("train_id", train_id), ("val_id", val_id), ("test_id", test_id)]:
        mx = int(max(split["labels"])) if len(split) > 0 else -1
        if mx >= NUM_ID_LABELS:
            raise ValueError(f"{name} contains label >= {NUM_ID_LABELS}. max={mx}")

    cols = ["input_ids", "attention_mask", "labels"]
    train_id.set_format(type="torch", columns=cols)
    val_id.set_format(type="torch", columns=cols)
    test_id.set_format(type="torch", columns=cols)
    val_ood.set_format(type="torch", columns=cols)
    test_ood.set_format(type="torch", columns=cols)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_id, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader   = torch.utils.data.DataLoader(val_id, batch_size=args.batch_size)
    test_loader  = torch.utils.data.DataLoader(test_id, batch_size=args.batch_size)
    ood_loader   = torch.utils.data.DataLoader(test_ood, batch_size=args.batch_size)

    if args.bayes == "svdkl":
        model = DistilBERT_SVDKL(num_labels=NUM_ID_LABELS).to(args.device)
        # Optimizer for feature extractor, GP hyper/variational params, and likelihood
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.proj.parameters(), "lr": args.lr_head},
            {"params": model.gp_layer.hyperparameters(), "lr": args.lr_head * 0.1},
            {"params": model.gp_layer.variational_parameters(), "lr": args.lr_head},
            {"params": model.likelihood.parameters(), "lr": args.lr_head},
        ])
        total_steps = args.epochs * len(train_loader)
        lr_sched = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.06 * total_steps),
            num_training_steps=total_steps,
        )
        mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp_layer, num_data=len(train_loader.dataset))
    else:
        model = DistilBERT_BNN(num_labels=NUM_ID_LABELS, dyadic_level=args.dyadic_level, bayes=args.bayes).to(args.device)
        # Optimizer groups
        head_params = list(model.proj.parameters()) + list(model.bayes.parameters())
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": args.lr_head},
            {"params": backbone_params, "lr": args.lr_backbone},
        ], weight_decay=0.01)

        total_steps = args.epochs * len(train_loader)
        kl_warmup_steps = int(args.kl_warmup_frac * total_steps)

        lr_sched = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.06 * total_steps),
            num_training_steps=total_steps,
        )

    dataset_size = len(train_id)
    global_step = 0

    # log training time
    train_start = time.time()
    last_val_acc = None
    last_val_nll = None
    last_val_ece = None

    # ----- Train -----
    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_start = time.time()

        if args.bayes != "svdkl":
            if epoch <= args.freeze_epochs:
                set_backbone_trainable(model, trainable=False)
            else:
                set_backbone_trainable(model, trainable=True, unfreeze_last_n_layers=2)

        losses = []
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}

            if args.bayes == "svdkl":
                model.train()
                model.likelihood.train()
                output = model(batch["input_ids"], batch["attention_mask"])
                loss = -mll(output, batch["labels"])
            else:
                logits, kl = model(batch["input_ids"], batch["attention_mask"])
                ce = F.cross_entropy(logits, batch["labels"])
                beta = beta_warmup(global_step, kl_warmup_steps)
                loss = ce + beta * (kl / dataset_size)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_sched.step()

            losses.append(loss.item())
            global_step += 1

        val_acc = compute_id_accuracy(model, val_loader, args.device)
        val_nll = compute_id_nll(model, val_loader, args.device)
        val_ece = compute_id_ece(model, val_loader, args.device)
        last_val_acc, last_val_nll, last_val_ece = val_acc, val_nll, val_ece
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch}: train_loss={np.mean(losses):.4f} val_acc(ID)={val_acc:.4f} "
            f"val_nll(ID)={val_nll:.4f} val_ece(ID)={val_ece:.4f} time={epoch_time:.1f}s"
        )

    # ----- ID test accuracy -----
    total_train_time = time.time() - train_start
    print(f"Total training time: {total_train_time/60:.2f} minutes")
    test_acc = compute_id_accuracy(model, test_loader, args.device)
    test_nll = compute_id_nll(model, test_loader, args.device)
    test_ece = compute_id_ece(model, test_loader, args.device)
    print(f"ID Test: acc={test_acc:.4f} nll={test_nll:.4f} ece={test_ece:.4f}")

    # ----- OOD detection using uncertainty on ID-test vs OOD-test -----
    all_is_ood = []
    all_ent = []
    all_mi = []

    def collect_scores(loader, is_ood_flag: int):
        nonlocal all_is_ood, all_ent, all_mi
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            probs_T, mean_probs = mc_probs(
                model,
                {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
                T=args.T
            )
            ent, mi = uncertainty_scores(probs_T, mean_probs)
            all_is_ood.extend([is_ood_flag] * len(ent))
            all_ent.extend(ent.tolist())
            all_mi.extend(mi.tolist())

    collect_scores(test_loader, is_ood_flag=0)   # ID
    collect_scores(ood_loader,  is_ood_flag=1)   # OOD

    y = np.array(all_is_ood, dtype=np.int32)
    s_ent = np.array(all_ent, dtype=np.float64)
    s_mi  = np.array(all_mi, dtype=np.float64)

    auroc_ent = roc_auc_score(y, s_ent)
    auprc_ent = average_precision_score(y, s_ent)
    auroc_mi  = roc_auc_score(y, s_mi)
    auprc_mi  = average_precision_score(y, s_mi)

    print(f"OOD (Entropy) AUROC={auroc_ent:.4f} AUPRC={auprc_ent:.4f}")
    print(f"OOD (MI)      AUROC={auroc_mi:.4f} AUPRC={auprc_mi:.4f}")

    metrics = {
        "seed": args.seed,
        "val_acc": last_val_acc,
        "val_nll": last_val_nll,
        "val_ece": last_val_ece,
        "test_acc": test_acc,
        "test_nll": test_nll,
        "test_ece": test_ece,
        "ood_auroc_ent": auroc_ent,
        "ood_auprc_ent": auprc_ent,
        "ood_auroc_mi": auroc_mi,
        "ood_auprc_mi": auprc_mi,
        "train_minutes": total_train_time / 60.0,
        "train_seconds": total_train_time,
    }

    if args.log_csv:
        fieldnames = list(metrics.keys())
        file_exists = os.path.exists(args.log_csv)
        with open(args.log_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
        print(f"Metrics appended to {args.log_csv}")

    return metrics


if __name__ == "__main__":
    main()
