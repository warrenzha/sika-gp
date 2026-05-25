import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from csgp.models.dgp import DeepCSGP


class DGPUCI(object):
    def __init__(self, args, in_features, hidden_features=None, batch_size=512, sparse=True) -> None:
        self.args = args
        self.in_features = in_features
        self.hidden_features = hidden_features if hidden_features is not None else [64, 64]
        self.batch_size = batch_size
        self.sparse = sparse

    def model_setup(self):
        args = self.args
        ###################################
        # Define Model
        ###################################
        model = DeepCSGP(
            in_features=self.in_features,
            out_features=1,
            hidden_features=self.hidden_features,
            dyadic_levels=[args.dyadic_level] * (len(self.hidden_features) + 1),
        )
        
        
        # model = torch.compile(model,mode='default') # Compile it
        criterion = torch.nn.GaussianNLLLoss()  # torch.nn.MSELoss()

        # enable cuda
        if torch.cuda.is_available():
            model.cuda()
            criterion.cuda()
        if args.half:
            model.half()
            criterion.half()

        self.model = model
        self.criterion = criterion

    @staticmethod
    def _target_column(target):
        return target.reshape(-1, 1)

    def train(self, train_loader, val_loader=None):

        args = self.args
        model = self.model
        criterion = self.criterion

        # switch to train mode
        model.train()

        # train
        train_losses = []
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        epochs_iter = tqdm(range(args.epochs), desc=f"Training {model.__class__.__qualname__}")
        minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
        for epoch in epochs_iter:
            running_loss = 0.0

            for (input, target) in minibatch_iter:

                # enable cuda
                input, target = input.to(args.device), target.to(args.device)
                target = self._target_column(target)
                optimizer.zero_grad()

                ############################################
                # Compute NLL by MC sampling
                ############################################
                output_ = []
                kl_ = []
                for mc_run in range(args.num_mc_train):
                    outputs, kls = model(input, return_kl=True, sparse=self.sparse)
                    output_.append(outputs)
                    kl_.append(kls)
                outputs = torch.stack(output_)
                kls = torch.stack(kl_)
                
                output_mean = torch.mean(outputs, dim=0)
                kl = torch.mean(kls, dim=0)
                # nll = criterion(
                #     outputs, target, args.noise_var ** 2 * torch.ones_like(outputs),
                # ).mean(dim=0)
                nll = F.mse_loss(output_mean, target)

                scaled_kl = kl / len(train_loader.dataset)
                loss = nll + scaled_kl  # ELBO loss

                # compute gradient and do SGD step
                loss.backward()
                optimizer.step()

                output_mean = output_mean.float()
                loss = loss.float()

                # record loss
                running_loss += loss.item() * input.size(0)

            epochs_iter.set_postfix(loss=loss.item())
            train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            # validate
            if args.validate:
                best_loss = float('inf')
                val_loss = self.validate(args, val_loader, model, criterion, self.sparse)
                if args.verbose:
                    print(f'Epoch {epoch + 1}\t\t',
                          f'Training Loss: {train_loss}\t\t',
                          f'Validation Loss: {val_loss}')
                if val_loss < best_loss:
                    best_loss = val_loss
                    filename = f'{model.__class__.__qualname__.lower()}_{args.dnn_name}_{train_loader.dataset.dataset.__name__}.pth'
                    save_path = os.path.join(args.checkpoint_dir, filename)
                    torch.save(model.state_dict(), save_path)

        self.train_loader = train_loader
        self.model = model
        self.train_losses = train_losses
        return model, train_losses

    @staticmethod
    def validate(args, val_loader, model, criterion, sparse):
        # switch to evaluate mode
        model.eval()

        # validate
        with torch.no_grad():
            running_loss = 0.0
            for (input, target) in val_loader:

                # enable cuda
                input, target = input.to(args.device), target.to(args.device)
                target = DGPUCI._target_column(target)

                ############################################
                # Compute NLL by MC sampling
                ############################################
                output_ = []
                kl_ = []
                for mc_run in range(args.num_mc_test):
                    outputs, kls = model(input, return_kl=True, sparse=sparse)
                    output_.append(outputs)
                    kl_.append(kls)
                outputs = torch.stack(output_)
                kls = torch.stack(kl_)
                    
                output_mean = torch.mean(outputs, dim=0)
                kl = torch.mean(kls, dim=0)

                nll = criterion(
                    outputs, target, args.noise_var ** 2 * torch.ones_like(outputs),
                )

                scaled_kl = kl / len(val_loader.dataset)
                loss = nll + scaled_kl  # ELBO loss

                output_mean = output_mean.float()
                loss = loss.float()

                # record loss
                running_loss += loss.item() * input.size(0)
            val_loss = running_loss / len(val_loader.dataset)
        return val_loss

    def test(self, test_loader, model):

        args = self.args

        # initialize
        means_list = []
        stds_list = []
        input_list = []
        target_list = []

        # switch to evaluate mode
        model.eval()

        # test
        with torch.no_grad():
            for (input, target) in test_loader:
                # enable cuda
                input, target = input.to(args.device), target.to(args.device)
                target = self._target_column(target)

                ############################################
                # Compute NLL by MC sampling
                ############################################
                output_ = []
                for mc_run in range(args.num_mc_test):
                    outputs, _ = model(input, return_kl=True, sparse=self.sparse)
                    output_.append(outputs)
                outputs = torch.stack(output_)

                means = outputs.mean(dim=0)
                stds = (outputs.var(dim=0, unbiased=False) + args.noise_var ** 2).sqrt()
                means_list.append(means)
                stds_list.append(stds)
                input_list.append(input)
                target_list.append(target)

            # get mean and std
            pred_mean = torch.cat(means_list)
            pred_std = torch.cat(stds_list)
            input_true = torch.cat(input_list)
            target_true = torch.cat(target_list)

        self.pred_mean = pred_mean
        self.pred_std = pred_std
        self.input_true = input_true
        self.target_true = target_true
        return pred_mean, pred_std, input_true, target_true
