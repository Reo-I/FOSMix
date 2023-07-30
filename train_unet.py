import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import source
from source.load_data import (
    dataset as customized_dataset,
    extract_data,
)
from source.models import fcsmix, networks
from source.training import losses, metrics, runner
from source.constant import OEM, FLAIR, _load_config
from source.utils import set_logging
from logging import getLogger

def main(args):
    # -----------------------
    # --- Main parameters ---
    # -----------------------
    config = _load_config()
    dataset = args.dataset
    seed = args.seed
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    optimize = bool(args.optimize)
    randomize = bool(args.randomize)
    if dataset == "OEM":
        #OpenEarthMap
        classes = OEM.classes
        label_to_anno = OEM.label_to_anno
        class_obj = OEM.class_obj
        
    elif dataset == "FLAIR":
        #FLAIR
        classes = FLAIR.classes
        label_to_anno = FLAIR.label_to_anno
        class_obj = FLAIR.class_obj

    n_classes = len(classes) + 1
    classes_wt = np.ones([n_classes], dtype=np.float32)
    pb_set = args.pb_set
    
    if args.prob_imagenet_ref<0 or args.prob_imagenet_ref >1:
        raise TypeError("set the probability of use ImageNet for reference from 0 to 1")
    elif args.prob_imagenet_ref>0 and args.server == "wisteria":
        logger.info("sample reference from ImageNet")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    outdir = "weights" + args.ver
    os.makedirs(outdir, exist_ok=True)
    results_img_dir = f"{outdir}/img"
    os.makedirs(results_img_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------
    # --- split training and validation sets ---
    # -------------------------------------------
    if args.server == "wisteria":
        path = config.data_dir["wisteria"] 
    elif args.server == "colab":
        path = config.data_dir["colab"]

    if dataset == "OEM":
        path = path + "openearthmap/"
        exd = extract_data.ExtractData(dataset=dataset, path=path)
        img_file, ano_file = exd.extract_fns_oem(pb_set=pb_set)
    
    elif dataset == "FLAIR":
        path = path + "flair/"
        exd = extract_data.ExtractData(dataset = dataset, path = path)
        img_file, ano_file = exd.extract_fns_flair()

    logger.info("Train samples : %s", len(img_file[0]))
    logger.info("Reference samples : %s", len(img_file[1]))
    logger.info("Test  samples : %s", len(img_file[2]))
    logger.info("Valid  samples : %s", len(img_file[3]))

    # ---------------------------
    # --- Define data loaders ---
    # ---------------------------
    trainset = customized_dataset.Dataset(img_file[0], ano_file[0], ref_list = img_file[1], \
        classes=classes, train=True, randomize = randomize, args = args)
    validset = customized_dataset.Dataset(img_file[3], ano_file[3], ref_list = [], \
        classes=classes, train=False, randomize = False, args = args, p_down_scale=0.0)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=8)

    testset = customized_dataset.Dataset(img_file[2], ano_file[2], ref_list = [], \
        classes=classes, train=False, randomize = False, args = args, p_down_scale = 1.0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    # --------------------------
    #       network setup
    # --------------------------
    fcmixnet = fcsmix.FCsMix(args = args)
    network = networks.load_network(args=args, n_classes=n_classes)
    if args.loss_type == "jaccard":
        criterion = losses.JaccardLoss(class_weights=classes_wt)
    elif args.loss_type == "bce":
        criterion = nn.CEWithLogitsLoss() #cross entropy でいい
    else:
        raise ValueError("loss function's name is NOT correct")

    metric = metrics.IoU()
    if optimize:
        optimizer_fcsmix = torch.optim.Adam(fcmixnet.parameters(), lr=learning_rate)
    optimizer_net = torch.optim.Adam(network.parameters(), lr=learning_rate)

    network_fout = f"{network.name}_s{seed}_{criterion.name}"
    logger.info("Model output name  : %s", network_fout)

    if torch.cuda.device_count() > 1:
        logger.info("Number of GPUs : %s", torch.cuda.device_count())

        fcmixnet = torch.nn.DataParallel(fcmixnet)
        network = torch.nn.DataParallel(network)
        if randomize and optimize:
            optimizer_fcsmix = torch.optim.Adam([dict(params=fcmixnet.module.parameters(), lr=learning_rate)])
        optimizer_net = torch.optim.Adam([dict(params=network.module.parameters(), lr=learning_rate)])

    if optimize:
        optimizers = {'fcsmix': optimizer_fcsmix, 'net': optimizer_net}
    else:
        optimizers = {'net': optimizer_net}

    # ------------------------
    # --- Network training ---
    # ------------------------
    start = time.time()
    best_loss = float("inf")
    best_metrix = 0.0
    train_hist = []
    valid_hist = []
    logs = []

    test_ious = {}
    
    for epoch in range(n_epochs):
        
        logger.info("\nEpoch: %s", epoch + 1)

        logs_train = runner.train_epoch(
            premodel = fcmixnet,
            model=network,
            optimizers=optimizers,
            criterion=criterion,
            metric=metric,
            dataloader=train_loader,
            epoch = epoch,
            n_epochs = n_epochs,
            n_classes = n_classes,
            device=device,
            args = args,
        )

        logs_valid = runner.valid_epoch(
            model=network,
            criterion=criterion,
            metric=metric,
            dataloader=valid_loader,
            device=device,
            n_classes=n_classes,
            args=args,
        )
        train_hist.append(logs_train)
        valid_hist.append(logs_valid)
        
        if (epoch+1>50) and ((epoch+1)%20 == 0):
            test_iou, _ = runner.test(
                model=network,
                metric=metrics.iou,
                n_classes = n_classes,
                dataloader=test_loader,
                device=device,
                args=args,
                l2a=label_to_anno,
                path = results_img_dir,
                class_obj=class_obj,
                logs = logs,
                epoch = epoch,
                )
            logger.info("test iou: %s", test_iou)
            logger.info("test miou: %s", np.mean(test_iou))
            test_ious[epoch+1] = np.mean(test_iou)


        source.utils.progress(
            train_hist,
            valid_hist,
            criterion.name,
            metric.name,
            n_epochs,
            outdir,
            network_fout,
            test_ious,
            args,
        )

        loss_value = logs_valid[criterion.name]
        mean_metric_value = np.mean(logs_valid[metric.name])
        if best_loss > loss_value:
            best_loss = loss_value
            torch.save(fcmixnet.state_dict(), os.path.join(outdir, "fcmixnet.pth"))
            torch.save(network.state_dict(), os.path.join(outdir, f"{network_fout}.pth"))
            logger.info("Model saved based on loss !")
        if best_metrix < mean_metric_value:
            best_metrix = mean_metric_value
            torch.save(fcmixnet.state_dict(), os.path.join(outdir, "fcmixnet_metrix.pth"))
            torch.save(network.state_dict(), os.path.join(outdir, f"{network_fout}_metrix.pth"))
            logger.info("Model saved based on matrix !")

    logger.info("Completed: %.4f min.", (time.time() - start) / 60.0)
    torch.save(network.state_dict(), os.path.join(outdir, f"{network_fout}_final.pth"))

    if optimize and not args.MFI:
        noise = torch.randn(1, 100, 1, 1).to(device)
        if torch.cuda.device_count() > 1:
            mask_for_save = fcmixnet.module.maskG(noise)
        else:
            mask_for_save = fcmixnet.maskG(noise)
        mask_for_save = mask_for_save.detach().cpu().numpy()
        np.save(outdir + '/mask', mask_for_save)

    learned_path = os.path.join(outdir, f"{network_fout}.pth")
    network.load_state_dict(torch.load(learned_path))

    testset = customized_dataset.Dataset(img_file[2], ano_file[2], ref_list = [], \
        classes=classes, train=False, final = True, args=args)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)

    test_iou, df_results = runner.final(
            model=network,
            metric=metrics.iou,
            n_classes=n_classes,
            dataloader=test_loader,
            device=device,
            args=args,
            l2a=label_to_anno,
            path = outdir,
            class_obj=class_obj,
            logs = logs,
            epoch = -1,
        )
    logger.info(test_iou)
    logger.info(np.mean(test_iou))
    df_results.to_csv(outdir +"/log_output.csv")
    logger.info("Save test results")

if __name__ == '__main__':
    base = source.options.BaseOptions()
    args = base.initialize()
    base.show_options(args)
    set_logging()
    logger = getLogger(__name__)
    main(args)
    