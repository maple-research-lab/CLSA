import time
import torch.nn as nn
import torch

from training.train_utils import AverageMeter,ProgressMeter,accuracy

def train(train_loader, model, criterion, optimizer, epoch, args,log_path):
    """
    :param train_loader:  data loader
    :param model: training model
    :param criterion: loss function
    :param optimizer: SGD optimizer
    :param epoch: current epoch
    :param args: config parameter
    :return:
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    mse_criterion=nn.MSELoss().cuda(args.gpu)
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            len_images = len(images)
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            crop_copy_length = int((len_images - 1) / 2)
            image_k = images[0]
            image_q = images[1:1 + crop_copy_length]
            image_strong = images[1 + crop_copy_length:]

        output, target, output2, target2 = model(image_q, image_k, image_strong)
        loss_contrastive = 0
        loss_weak_strong = 0
        if epoch == 0 and i == 0:
            print("-" * 100)
            print("contrastive loss count %d" % len(output))
            print("weak strong loss count %d" % len(output2))
            print("-" * 100)
        for k in range(len(output)):
            loss1 = criterion(output[k], target[k])
            loss_contrastive += loss1
        for k in range(len(output2)):
            loss2 = -torch.mean(torch.sum(torch.log(output2[k]) * target2[k], dim=1))  # DDM loss
            loss_weak_strong += loss2
        loss = loss_contrastive + args.alpha * loss_weak_strong
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output[0], target[0], topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            progress.write_record(i,log_path)
    return top1.avg