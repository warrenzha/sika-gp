import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score

BACKBONE = "distilbert-base-uncased"

# ----------------- BNN modules -----------------
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

    def forward(self, x):
        w_sigma = self._sigma(self.w_rho)
        b_sigma = self._sigma(self.b_rho)

        w = self.w_mu + w_sigma * torch.randn_like(self.w_mu)
        b = self.b_mu + b_sigma * torch.randn_like(self.b_mu)

        logits = F.linear(x, w, b)
        kl = self.kl_div(w_sigma, b_sigma)
        return logits, kl

    def kl_div(self, w_sigma, b_sigma):
        prior_var = self.prior_sigma ** 2

        def kl_gauss(mu, sigma):
            var = sigma ** 2
            return 0.5 * torch.sum((var + mu**2) / prior_var - 1.0 + torch.log(prior_var / var))

        return kl_gauss(self.w_mu, w_sigma) + kl_gauss(self.b_mu, b_sigma)


class DistilBERT_BNN(nn.Module):
    def __init__(self, num_labels=2, proj_dim=256, prior_sigma=1.0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(BACKBONE)
        hidden = self.backbone.config.hidden_size  # 768
        self.proj = nn.Linear(hidden, proj_dim)
        self.bayes = BayesianLinear(proj_dim, num_labels, prior_sigma=prior_sigma)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        z = out.last_hidden_state[:, 0, :]  # CLS
        z = self.proj(z)
        logits, kl = self.bayes(z)
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

        # Always allow layer norm / embeddings? (optional) keep frozen for stability
        # for p in model.backbone.embeddings.parameters(): p.requires_grad = True


@torch.no_grad()
def mc_predict(model, batch, T=30):
    model.eval()
    probs = []
    for _ in range(T):
        logits, _ = model(**batch)
        probs.append(torch.softmax(logits, dim=-1))
    probs = torch.stack(probs, dim=0)  # (T,B,C)
    mean_probs = probs.mean(dim=0)
    entropy = -(mean_probs * mean_probs.clamp_min(1e-9).log()).sum(dim=-1)

    ent_each = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)  # (T,B)
    mi = entropy - ent_each.mean(dim=0)
    return mean_probs, entropy, mi


def beta_warmup(step, warmup_steps):
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=2e-5)
    ap.add_argument("--kl_warmup_frac", type=float, default=0.15)
    ap.add_argument("--freeze_epochs", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ds = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)

    def tok(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

    ds = ds.map(tok, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = torch.utils.data.DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds["validation"], batch_size=args.batch_size)

    model = DistilBERT_BNN(num_labels=2).to(args.device)

    # Optim groups
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

    dataset_size = len(ds["train"])
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Freeze backbone early, then unfreeze last 2 layers
        if epoch <= args.freeze_epochs:
            set_backbone_trainable(model, trainable=False)
        else:
            set_backbone_trainable(model, trainable=True, unfreeze_last_n_layers=2)

        # Need to rebuild optimizer param list for backbone if you change requires_grad? Keep simple:
        # We'll just leave optimizer; frozen params won't get grads anyway.

        losses = []
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
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

        # Validation (deterministic = single sample)
        model.eval()
        y_true, y_pred = [], []
        val_nll = []
        for batch in val_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            logits, _ = model(batch["input_ids"], batch["attention_mask"])
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            y_true.extend(batch["labels"].cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            val_nll.append(F.nll_loss(probs.log(), batch["labels"]).item())

        acc = accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch}: train_loss={np.mean(losses):.4f} val_acc={acc:.4f} val_nll={np.mean(val_nll):.4f}")

    # Example MC uncertainty on a few val samples
    b = next(iter(val_loader))
    b = {k: v.to(args.device) for k, v in b.items()}
    mean_probs, entropy, mi = mc_predict(model, {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}, T=30)
    print("MC entropy (first 5):", entropy[:5].detach().cpu().numpy())
    print("MC MI (first 5):", mi[:5].detach().cpu().numpy())


if __name__ == "__main__":
    main()
