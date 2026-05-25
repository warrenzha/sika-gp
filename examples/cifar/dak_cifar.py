import torch
import torch.nn.functional as F

from csgp.models.dak import DAK
import csgp.models.dnn.resnet_large as resnet
from csgp.utils import ece_score, accuracy
from csgp.utils import AverageMeter


def warmup_lr_lambda(epoch):
    if epoch < 5:  # Assuming a 5-epoch warm-up period
        return epoch / 5
    else:
        return 1


class DAKCIFAR:
    def __init__(self, args):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        print("Using: ", self.device)

        torch.manual_seed(args.seed)
        
        self.num_classes = args.num_classes
        self.epochs = args.epochs
        self.log_interval = args.log_interval
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_mc_train = args.num_mc_train
        self.num_mc_test = args.num_mc_test
        self.sparse = args.use_sparse
        
        feature_extractor = resnet.__dict__[args.arch](num_classes=args.num_classes, classifier=False)
        model = DAK(
            feature_extractor=feature_extractor,
            num_features=args.num_features,
            num_tasks=args.num_classes,
            dyadic_level=args.dyadic_level,
            ell_c=0.5,
        )
            
        if args.pretrained:
            if args.num_classes == 10:
                model.feature_extractor.load_state_dict(
                    torch.load('./checkpoint/nn_cifar_10.pth')['state_dict'],
                    strict=False
                )
            else:
                model.feature_extractor.load_state_dict(
                    torch.load('checkpoint/nn_cifar_100.pth')['state_dict'],
                    strict=False
                )

        self.model = model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                                         weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_lr_lambda)

        if args.fix_features and args.pretrained:
            for name, params in self.model.named_parameters():
                if "feature_extractor" in name:
                    params.requires_grad = False

    def reset_optimizer(self, epoch):
        lr = self.lr * (0.1 ** (epoch // 20))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, train_loader, epoch):
        self.model.train()

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # input_var = torch.cat([data for _ in range(self.num_mc_train)], 0)
            # output_mc, kl = self.model(input_var, sparse=self.sparse)
            # output = output_mc.reshape(self.num_mc_train, -1, self.num_classes).mean(dim=0)
            # loss = F.cross_entropy(output, target) + kl / self.batch_size  # ELBO loss
            
            output, kl = self.model.forward_with_MC(data, num_mc=self.num_mc_train, sparse=self.sparse)
            loss = F.cross_entropy(output, target) + kl / self.batch_size  # ELBO loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))

            if batch_idx % self.log_interval == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch,
                    batch_idx,
                    len(train_loader),
                    loss=losses,
                    top1=top1))

        if epoch <= 5:
            self.warmup_scheduler.step()

        return losses.avg, top1.avg

    def test(self, test_loader, ece_bins=10):
        self.model.eval()
        print("Length scale: ", self.model.gp.ell_c)
        correct = 0
        nll = 0
        ece = 0
        batch_count = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # input_var = torch.cat([data for _ in range(self.num_mc_train)], 0)
                # output_mc, _ = self.model(input_var, sparse=self.sparse)
                # output = output_mc.reshape(self.num_mc_train, -1, self.num_classes).mean(dim=0)

                output, _ = self.model.forward_with_MC(data, num_mc=self.num_mc_test, sparse=self.sparse)
                
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()
                nll += F.cross_entropy(output, target, reduction='sum').item()
                ece += ece_score(probs.cpu().numpy(), target.cpu().numpy(), n_bins=ece_bins)
                batch_count += 1

        acc = 100. * correct / len(test_loader.dataset)
        nll /= len(test_loader.dataset)
        ece /= batch_count

        return acc, nll, ece

    def validate(self, val_loader):
        self.model.eval()
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        nll = AverageMeter('NLL', ':.4e')
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # input_var = torch.cat([data for _ in range(self.num_mc_train)], 0)
                # output_mc, kl = self.model(input_var, sparse=self.sparse)
                # output = output_mc.reshape(self.num_mc_train, -1, self.num_classes).mean(dim=0)
                
                output, kl = self.model.forward_with_MC(data, num_mc=self.num_mc_test, sparse=self.sparse)
                
                val_loss = F.cross_entropy(output, target) + kl / 1000  # ELBO loss

                output = output.float()
                loss = val_loss.float()
                neg_log_likelihood = F.cross_entropy(output, target)

                # measure accuracy and record loss
                prec1 = accuracy(output.data, target)[0]
                losses.update(loss.item(), data.size(0))
                top1.update(prec1.item(), data.size(0))
                nll.update(neg_log_likelihood.item(), data.size(0))

        print(
            '\nValidation set: Average loss: {:.4f},  Prec@1: {}/{} ({:.2f}%)\n'.format(
                losses.avg, top1.sum / 100, len(val_loader.dataset),
                top1.avg))

        return top1.avg, nll.avg


if __name__ == '__main__':
    resnet_18_model = resnet.resnet18(num_classes=10, classifier=False)
    resnet_34_model = resnet.resnet34(num_classes=100, classifier=False)

    model_10 = DAK(
        feature_extractor=resnet_18_model,
        num_tasks=10,
        num_features=64,
        dyadic_level=7,
        grid_bounds=(0., 1.),
    )

    model_100 = DAK(
        feature_extractor=resnet_34_model,
        num_tasks=100,
        num_features=128,
        dyadic_level=7,
        grid_bounds=(0., 1.),
    )

    total_resnet18_params = sum(p.numel() for p in model_10.parameters())
    print("Total resnet 18 parameters: ", total_resnet18_params)

    total_resnet34_params = sum(p.numel() for p in model_100.parameters())
    print("Total resnet 34 parameters: ", total_resnet34_params)
