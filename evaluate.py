""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
from visualize import plot


config = SearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data, test_dat, val_dat = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=True,validation2 = True)

    net_crit = nn.CrossEntropyLoss().to(device)
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    model = model.to(device)

    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    #balanced split to train/validation
    print(train_data)
    
    # split data to train/validation
    n_train = len(train_data)
    n_val = len(val_dat)
    n_test = len(test_dat)
    split = n_train // 2
    indices1 = list(range(n_train))
    indices2 = list(range(n_val))
    indices3 = list(range(n_test))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices1)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices2)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices3)
    
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_dat,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dat,
                                               batch_size=config.batch_size,
                                               sampler=test_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    
    #load
    if(config.load):
        model,config.epochs,w_optim,alpha_optim,net_crit = utils.load_checkpoint(model,config.epochs,w_optim,alpha_optim,net_crit,'/content/pt.darts/searchs/custom/checkpoint.pth.tar')
     
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, config.w_momentum, config.w_weight_decay)
    
    #model  = torch.load('/content/pt.darts/searchs/custom/checkpoint.pth.tar')
       
    #print("Loaded!")
    # training loop
    best_top1 = 0.
    best_top_overall = -999
    config.epochs = 300#BUG, config epochs ta com algum erro
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)
        
        print("###################TRAINING#########################")
        # training
        train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch)
        print("###################END TRAINING#########################")
        
        # validation
        cur_step = (epoch+1) * len(train_loader)
        print("###################VALID#########################")
        top_overall = validate(valid_loader, model, epoch, cur_step,overall = True)
        print("###################END VALID#########################")
        
        # test
        print("###################TEST#########################")
        top1 = validate(test_loader, model, epoch, cur_step)
        print("###################END TEST#########################")
        
        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        #save best overall(macro avg of f1 prec and recall)
        if(best_top_overall < top_overall):
            best_top_overall = top_overall
            best_genotype_overall  = genotype
            is_best_overall = True
        else:
            is_best_overall = False
        
        utils.save_checkpoint(model,epoch,w_optim,alpha_optim,net_crit, config.path, is_best,is_best_overall)
        print("saved!")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
    logger.info("Best Genotype Overall = {}".format(best_genotype_overall))


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, epoch, cur_step,overall = False):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.eval()
    import numpy as np
    preds = np.asarray([])
    targets = np.asarray([])
    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            target = y
            output = logits
            topk = (1,3)
            maxk = max(topk)
            batch_size = target.size(0)
            _, predicted = torch.max(output.data, 1)
            #minha alteracao
            preds = np.concatenate((preds,predicted.cpu().numpy().ravel()))
            targets = np.concatenate((targets,target.cpu().numpy().ravel()))
            
            ###TOP 5 NAO EXISTE NAS MAAMAS OU NO GEO. TEM QUE TRATAR
            maxk = 3 # Ignorando completamente o top5
            
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            
            # one-hot case
            if target.ndimension() > 1:
                target = target.max(1)[1]

            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0)
                res.append(correct_k.mul_(1.0 / batch_size))
            prec1,prec5 = res
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

    
            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))
            
    print(preds.shape)
    print(targets.shape)
    print('np.unique(targets):',np.unique(targets))
    print('np.unique(preds): ',np.unique(preds))
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    print(accuracy_score(targets, preds))
    cr = classification_report(targets, preds,output_dict= True)
    a1,a2,a3 = cr['macro avg']['f1-score'] ,cr['macro avg']['precision'],cr['macro avg']['recall'] 
    topover = (a1+a2+a3)/3 
    print(classification_report(targets, preds))
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import accuracy_score
    print(balanced_accuracy_score(targets, preds))
    print(accuracy_score(targets, preds))
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(targets, preds)
    print(matrix.diagonal()/matrix.sum(axis=1))
    print(matrix)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    logger.info("Valid: [{:2d}/{}] Overall {:.4%}".format(epoch+1, config.epochs, topover))
    
    if overall:
        return topover
    return top1.avg


if __name__ == "__main__":
    main()
