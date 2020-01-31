"""
Train a neural network on an image folder. Based on the PyTorch ImageNet example.
"""
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tensorboardX import SummaryWriter

import networks

model_names = sorted(name for name in networks.__dict__
    if name.islower() and not name.startswith("__")
    and callable(networks.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=-1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop-lr', '--drop-learning-rate', default=10, type=int,
                    metavar='N', help='drop learning rate after this many epochs (default: 10)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('--log', '-l', metavar='FILE', default=None,
                    help='filename to write the log to')
parser.add_argument('--attach', action='store_true',
                    help='attach a word recognition classifier')

best_prec1 = 0
device = torch.device('cpu')


def main():
    global args, best_prec1, device, summary
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.log is not None:
        print('Writing tensorboard log to', args.log)
        summary = SummaryWriter(args.log)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        args.data + '/train',
        transforms.Compose([
            #transforms.ToPILImage(mode='RGB'),
            #transforms.RandomResizedCrop(224) if args.arch == 'alexnet' else transforms.RandomResizedCrop(60),
            transforms.CenterCrop(60),
            transforms.ToTensor(),
            normalize,
        ]))
    target_num_classes = len(train_dataset.classes)

    # create model
    print("=> creating model '{}'".format(args.arch))
    #model = networks.__dict__[args.arch](num_classes=target_num_classes)
    model = networks.__dict__[args.arch](num_classes=553)

    if args.gpu is not None:
        device = torch.device(args.gpu)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("=> using device", device)
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if(args.start_epoch) == -1:
                args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            current_num_classes = num_classes(model)
            if current_num_classes != target_num_classes:
                print("=> changing number of classes from %d to %d" % (current_num_classes, target_num_classes))
                model = attach_classifier(model, target_num_classes).cuda()
            elif args.attach:
                print("=> attaching word classifier")
                model = attach_classifier(model, target_num_classes).cuda()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val',
            transforms.Compose([
                #transforms.ToPILImage(mode='RGB'),
                transforms.Resize(256) if args.arch == 'alexnet' else transforms.Resize(64),
                transforms.CenterCrop(224) if args.arch == 'alexnet' else transforms.CenterCrop(60),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(0 if args.start_epoch == -1 else args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        print('|| 1 ||')

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

        print('|| 2 ||')


def train(train_loader, model, criterion, optimizer, epoch):
    print("=> training on device", device)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))

        # Write tensorboard stuff
        if args.log is not None:
            summary.add_scalar('loss', losses.val, i)
            summary.add_scalar('Prec@1', top1.val, i)
            summary.add_scalar('Prec@5', top5.val, i)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every drop_lr epochs"""
    lr = args.lr * (0.1 ** (epoch // args.drop_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Visual(nn.Module):
    def __init__(self, base_model, layer=27, num_classes=200):
        super().__init__()

        # Copy features from base model
        #modulelist = list(base_model.features.modules())
        #self.features = nn.Sequential(*modulelist[1:layer+2])
        #for p in self.features.parameters():
        #    p.requires_grad = False
        self.features = base_model.features
        #for p in self.features.parameters():
        #    p.requires_grad = False

        # classifier1 = nn.Linear(4608, 4096)
        # classifier2 = nn.Linear(4096, 4096)
        # classifier3 = nn.Linear(4096, num_classes)
        # nn.init.normal_(classifier1.weight, 0, 0.01)
        # nn.init.constant_(classifier1.bias, 0)
        # nn.init.normal_(classifier2.weight, 0, 0.01)
        # nn.init.constant_(classifier2.bias, 0)
        # nn.init.normal_(classifier3.weight, 0, 0.01)
        # nn.init.constant_(classifier3.bias, 0)
        #
        # self.classifier = nn.Sequential(
        #     classifier1,
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     classifier2,
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     classifier3,
        # )


        # Attach new classifier
        modulelist = list(base_model.classifier.modules())[1:]
        #modulelist[0].requires_grad=False
        #for module in modulelist:
        #    module.requires_grad = False
        modulelist.pop()
        classifier3 = nn.Linear(4096, num_classes)
        nn.init.normal_(classifier3.weight, 0, 0.01)
        nn.init.constant_(classifier3.bias, 0)
        modulelist.append(classifier3)
        self.classifier = nn.Sequential(*modulelist)

        print(self)

    def forward(self, X):
        out = self.features(X)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


def attach_classifier(model, n):
    return Visual(model, num_classes=n)

def num_classes(model):
    final_layer = list(model.classifier.modules())[-1]
    num_classes = final_layer.weight.shape[0]
    return num_classes

if __name__ == '__main__':
    main()
