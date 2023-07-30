import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import cv2
import pathlib
import logging

from source.utils import save_predicted_img, show_test_result
from source.constant import BatchData

logger = logging.getLogger(__name__)
np.set_printoptions(suppress=True, precision=3, floatmode='fixed')

class AverageMeter:
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
        self.sum += val *n
        self.count += n
        self.avg = self.sum / self.count

def format_logs(logs):
    str_logs = ["{}={:.3}".format(k, v) for k, v in logs.items()]
    return ", ".join(str_logs)

def train_epoch(
    premodel=None,
    model=None,
    optimizers=None,
    criterion=None,
    metric=None,
    dataloader=None,
    epoch=None,
    n_epochs=None,
    n_classes=None,
    device="cpu",
    args = None, 
):
    
    loss_meter = AverageMeter()
    mask_loss_meter = AverageMeter()
    score_meter = np.zeros(n_classes-1)
    n_non_class = np.zeros(n_classes-1)
    logs = {}

    n_gpus = torch.cuda.device_count()

    premodel.to(device).train()
    model.to(device).train()

    max_iter = len(dataloader) * n_epochs
    n_iteration = len(dataloader) * epoch


    for i, sample in enumerate(dataloader):

        x = sample["x"].to(device, dtype=torch.float)
        y = sample["y"].to(device)
        n = x.shape[0]

        if n <n_gpus:
            logger.info("skip %s", n)
            continue
        if args.optimize:
            optimizers["fcsmix"].zero_grad()
        optimizers["net"].zero_grad()

        #randomize_batch = np.random.rand() < opt.randomize_prob
        if args.randomize:  # and randomize_batch:
            color_x = sample["color_x"].to(device, dtype=torch.float)
            ref = sample["ref"].to(device, dtype=torch.float)
            opt_mixed_x, full_mixed_x, msk = premodel.forward(color_x, ref)
            
            if args.optimize:
                opt_mixed_y = model.forward(torch.rot90(opt_mixed_x, 1, [2, 3]))
                opt_mixed_y = torch.rot90(opt_mixed_y, 3, [2, 3])
            if args.fullmask or args.halfmask:
                full_mixed_y = model.forward(full_mixed_x)

        outputs = model.forward(x)
        loss = criterion(outputs, y)
        if args.randomize:
            if args.optimize:
                loss += criterion(opt_mixed_y, y)
                lasso = args.LAMBDA1 * torch.mean(msk)
                var_mask = args.LAMBDA2 / torch.var(msk) 
                loss += lasso + var_mask

            if args.fullmask or args.halfmask:
                loss += criterion(full_mixed_y, y)
        
        if not args.COF:
            loss.backward()
        else:
            loss.backward(retain_graph=True)
            
        if args.optimize:
            optimizers["fcsmix"].step()
            
            if  (args.fullmask or args.halfmask) and args.COF:
                loss_cl = args.LAMBDA3 * F.l1_loss(opt_mixed_y, full_mixed_y, reduction='mean')
                if args.COF_all:
                    loss_cl += args.LAMBDA3 * F.l1_loss(opt_mixed_y, outputs, reduction='mean')
                    loss_cl += args.LAMBDA3 * F.l1_loss(full_mixed_y, outputs, reduction='mean')
                loss_cl.backward(inputs=list(model.parameters()))
        optimizers["net"].step()

        loss_meter.update(loss.cpu().detach().numpy(), n=n)

        if args.randomize and args.optimize:  # and randomize_batch:
            mask_loss_meter.update(lasso.cpu().detach().numpy() + var_mask.cpu().detach().numpy(), n=n)
            logs.update({"mask": mask_loss_meter.avg})
        
        scores, num_not = metric(outputs, y)
        score_meter +=scores
        n_non_class += num_not
        #score_meter.update(metric(outputs, y).cpu().detach().numpy())

        logs.update({criterion.name: loss_meter.avg})
        logs.update({metric.name: score_meter/n_non_class})

        #learning rate (poly learning rate)
        #------------------------------------------------------------
        multiply_lr = pow((1 - 1.0 * (n_iteration+1) / max_iter), 0.9)
        optimizers["net"].param_groups[0]["lr"] = args.learning_rate * multiply_lr
        if args.randomize and args.optimize:
            optimizers["fcsmix"].param_groups[0]["lr"] = args.learning_rate * multiply_lr
        n_iteration+=1
        #------------------------------------------------------------

    logger.info("Train IoU: %s", score_meter/n_non_class)
    return logs


def valid_epoch(
    model=None,
    criterion=None,
    metric=None,
    dataloader=None,
    device="cpu",
    n_classes=None,
    args = None
):

    loss_meter = AverageMeter()
    score_meter = np.zeros(n_classes-1)
    n_non_class = np.zeros(n_classes-1)
    n_gpus = torch.cuda.device_count()
    logs = {}
    model.to(device).eval()
    
    #with tqdm(dataloader, desc="Valid") as iterator:
    for sample in dataloader:
        x = sample["x"].to(device, dtype=torch.float)
        y = sample["y"].to(device)
        n = x.shape[0]
        if n <n_gpus:
            logger.info("skip %s", n)
            continue

        with torch.no_grad():
            outputs = model.forward(x)
            loss = criterion(outputs, y)
        loss_meter.update(loss.cpu().detach().numpy(), n=n)
        scores, num_not = metric(outputs, y)
        score_meter +=scores
        n_non_class += num_not

        logs.update({criterion.name: loss_meter.avg})
        logs.update({metric.name: score_meter/n_non_class})
    logger.info("Valid IoU: %s", score_meter/n_non_class)
    return logs

def test(
    model=None,
    metric=None,
    n_classes=None,
    dataloader=None,
    device="cpu",
    args = None, 
    l2a = None, 
    path = None, 
    class_obj = None, 
    logs = None, 
    epoch = None, 
):
    test_iou = np.zeros(n_classes-1)
    test_n = np.zeros(n_classes-1)
    classes = list(range(1, n_classes))
    uniform_data = np.zeros((n_classes-1, n_classes-1))
    n_gpus = torch.cuda.device_count()

    model.to(device).eval()

    for sample in dataloader:
        x = sample["x"].to(device, dtype=torch.float)
        y_gt = sample["y"].argmax(axis=1)
        h, w = sample["shape"]
        n = x.shape[0]
        if n < n_gpus:
            logger.info("skip %s", n)
            continue
        
        with torch.no_grad():
            pred = model.forward(x).cpu().detach().numpy()
        y_pr = pred.argmax(axis=1).astype("uint8")

        for b in range(y_pr.shape[0]):
            each_iou = np.zeros(n_classes-1)
            each_n = np.zeros(n_classes-1)
            for i in range(1, n_classes):
                y_pr_ = torch.Tensor(y_pr[b]) == i
                y_gt_ = y_gt[b] == i
                #calc iou
                each_n[i-1] = y_gt_.any()
                each_iou[i-1] = metric(y_pr_.float(), y_gt_.float(), for_metric = True)
        
                if epoch+1 == 0:
                    #for the last
                    #count gt/ pr
                    uni, count = np.unique(y_pr[b, y_gt_], return_counts = True)
                    uniform_data[i-1, uni[uni>0]-1] += count[uni>0]

            test_n += each_n
            test_iou+= each_iou
            rgb_fname = pathlib.Path(sample["fn"][0]).stem
            log = dict(zip(np.array(list(class_obj.values()))[1:][each_n>0.5] , each_iou[each_n>0.5]))
            log["fn"] = rgb_fname
            log["epoch"] = epoch+1 if epoch+1 !=0 else "final"
            log["domain"] = sample["domain"][b]
            logs.append(log)

    df_results = pd.DataFrame(logs, columns=["epoch"] + list(class_obj.values())[1:] + ["fn", "domain"])
    if epoch +1 == 0:
        uniform_data = uniform_data/uniform_data.sum(axis = 1)[:,None]
        show_test_result(uniform_data, list(class_obj.values())[1:], path)

    return test_iou/test_n, df_results

def final(
    model=None,
    metric=None,
    n_classes=None,
    dataloader=None,
    device="cpu",
    args = None,
    l2a = None,
    path = None,
    class_obj = None,
    logs = None,
    epoch = None,
):
    test_iou = np.zeros(n_classes-1)
    test_n = np.zeros(n_classes-1)
    classes = list(range(1, n_classes))
    uniform_data = np.zeros((n_classes-1, n_classes-1))
    
    model.to(device).eval()
    for sample in dataloader:
        x = sample["x"][0].to(device, dtype=torch.float)
        y_gt = sample["y"][0]
        h, w = sample["shape"]

        with torch.no_grad():
            msk = model(x)
            msk = msk.cpu().numpy()
            pred = (msk[0, :, :, :] + msk[1, :, :, ::-1] + msk[2, :, ::-1, :] + msk[3, :, ::-1, ::-1])/4
        y_pr = pred.argmax(axis=0).astype("uint8")
        if not args.test_crop:
            y_pr = cv2.resize(y_pr, (int(w[0]), int(h[0])), interpolation=cv2.INTER_NEAREST)

        each_iou = np.zeros(n_classes-1)
        each_n = np.zeros(n_classes-1)
        for i in range(1, n_classes):
            y_pr_ = torch.Tensor(y_pr) == i
            y_gt_ = y_gt == i
            #calc iou
            each_n[i-1] = y_gt_.any()
            each_iou[i-1] = metric(y_pr_.float(), y_gt_.float(), for_metric = True)
    
            if epoch+1 == 0:
                #for the last
                #count gt/ pr
                uni, count = np.unique(y_pr[y_gt_], return_counts = True)
                uniform_data[i-1, uni[uni>0]-1] += count[uni>0]

        test_n += each_n
        test_iou+= each_iou
        #save the predicted img
        if (epoch+1 ==0) and (not args.test_crop):
            rgb_fname = save_predicted_img(sample, y_gt, y_pr, classes, l2a, path = path+"/img")
        else:
            rgb_fname = pathlib.Path(sample["fn"][0]).stem
        #rgb_fname = pathlib.Path(sample["fn"][0]).stem
            
        log = dict(zip(np.array(list(class_obj.values()))[1:][each_n>0.5] , each_iou[each_n>0.5]))
        log["fn"] = rgb_fname
        log["epoch"] = epoch+1 if epoch+1 !=0 else "final"
        log["domain"] = sample["domain"][0]
        logs.append(log)
    df_results = pd.DataFrame(logs, columns=["epoch"] + list(class_obj.values())[1:] + ["fn", "domain"])
    if epoch +1 == 0:
        uniform_data = uniform_data/uniform_data.sum(axis = 1)[:,None]
        show_test_result(uniform_data, list(class_obj.values())[1:], path)

    return test_iou/test_n, df_results
