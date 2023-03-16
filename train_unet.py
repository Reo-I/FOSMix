#from cgi import test
from cmath import log
import os
import time
import sys
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
import source



def main(args):
    # -----------------------
    # --- Main parameters ---
    # -----------------------
    dataset = args.dataset
    seed = args.seed
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    optimize = bool(args.optimize)
    randomize = bool(args.randomize)
    if dataset == "OEM":
        #OpenEarthMap
        classes = [1, 2, 3, 4, 5, 6, 7, 8]
        label_to_anno = {0: [0, 0, 0], 1: [128,   0,   0], 2:[0, 255, 36], \
                    3:[148, 148, 148], 4:[255, 255, 255] , 5: [34, 97, 38], \
                    6 :[  0,  69, 255], 7: [ 75, 181,  73], 8: [222,  31,   7]}
        class_obj = {0:"None", 1: "bareland", 2:"grass", 3: "pavement", \
                4:"road", 5: "tree", 6: "water", 7:"cropland", 8: "building"}
        # 0: backgound, 1: bareland, 2: grass, 3: pavement
        # 4: road, 5: tree, 6: water, 7: cropland, 8: building 
        
    elif dataset == "FLAIR":
        #Flair
        classes = list(range(1, 13))
        """
        label_to_anno = ...
        class_obj = ...
        """

    n_classes = len(classes) + 1
    classes_wt = np.ones([n_classes], dtype=np.float32)
    pb_set = args.pb_set
    
    if args.prob_imagenet_ref<0 or args.prob_imagenet_ref >1:
        raise TypeError("set the probability of use ImageNet for reference from 0 to 1")
    elif args.prob_imagenet_ref>0 and args.server == "wisteria":
        print("sample reference from ImageNet")

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
    phase = ["train", "ref", "test", "val"]
    img_file = [[] for _ in range(len(phase))]
    ano_file = [[] for _ in range(len(phase))]
    phase_domains = [[] for _ in range(len(phase))]

    if dataset == "OEM":
        path = "/work/gu15/k36090/openearthmap/*/images/*.tif"
        pb_name = 'split_file/' if pb_set == "region" else 'split_file_for_resolution/'

        if not os.path.isfile(f"./{pb_name}train_path_fns.txt"):
            print(f"making file path for {pb_set}")
            source.extract_data.extract_fns(path, pb_name)

        for p in range(len(phase)):
            with open(pb_name+phase[p]+"_path_fns.txt", "r") as f:
                img_file[p] =  f.read().split("\n")[:-1]
            ano_file[p] = list(map(lambda x: x.replace('images', 'labels'), img_file[p]))
    
    elif dataset == "FLAIR":
        path = "/work/gu15/k36090/flair/"
        exd = source.extract_data.ExtractData(dataset = dataset, path = path)
        img_file, ano_file = exd.extract_fns_flair()


    print("Train samples :", len(img_file[0]))
    print("Reference samples :", len(img_file[1]))
    print("Test  samples :", len(img_file[2]))
    print("Valid  samples :", len(img_file[3]))

    # ---------------------------
    # --- Define data loaders ---
    # ---------------------------
    trainset = source.dataset.Dataset(img_file[0], ano_file[0], ref_list = img_file[1], \
        classes=classes, train=True, randomize = randomize, args = args)
    validset = source.dataset.Dataset(img_file[3], ano_file[3], ref_list = [], \
        classes=classes, train=False, randomize = False, args = args, p_down_scale=0.0)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=8)

    testset = source.dataset.Dataset(img_file[2], ano_file[2], ref_list = [], \
        classes=classes, train=False, randomize = False, args = args, p_down_scale = 1.0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    # --------------------------
    #       network setup
    # --------------------------
    fcmixnet = source.fcsmix.FCsMix(args = args)
    # network = source.unet.UNet(in_channels=3*args.n_input_channels, classes=n_classes)

    network = source.networks.load_network(args=args)
    if args.loss_type == "jaccard":
        criterion = source.losses.JaccardLoss(class_weights=classes_wt)
    elif args.loss_type == "bce":
        criterion = nn.CEWithLogitsLoss() #cross entropy でいい
    else:
        raise ValueError("loss function's name is NOT correct")


    metric = source.metrics.IoU()
    #params = [fcmixnet.parameters(), network.parameters()]
    #optimizer = torch.optim.Adam(itertools.chain(*params), lr=learning_rate)
    if optimize:
        optimizer_fcsmix = torch.optim.Adam(fcmixnet.parameters(), lr=learning_rate)
    optimizer_net = torch.optim.Adam(network.parameters(), lr=learning_rate)

    network_fout = f"{network.name}_s{seed}_{criterion.name}"
    print("Model output name  :", network_fout)

    if torch.cuda.device_count() > 1:
        print("Number of GPUs :", torch.cuda.device_count())

        fcmixnet = torch.nn.DataParallel(fcmixnet)
        network = torch.nn.DataParallel(network)
        #params = [fcmixnet.module.parameters(), network.module.parameters()]
        #optimizer = torch.optim.Adam([dict(params=itertools.chain(*params), lr=learning_rate)]
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
        
        print(f"\nEpoch: {epoch + 1}")

        logs_train = source.runner.train_epoch(
            premodel = fcmixnet,
            model=network,
            optimizers=optimizers,
            criterion=criterion,
            metric=metric,
            dataloader=train_loader,
            epoch = epoch,
            n_epochs = n_epochs,
            device=device,
            args = args,
        )

        logs_valid = source.runner.valid_epoch(
            model=network,
            criterion=criterion,
            metric=metric,
            dataloader=valid_loader,
            device=device,
            args=args, 
        )

        train_hist.append(logs_train)
        valid_hist.append(logs_valid)

        
        if (epoch+1>50) and ((epoch+1)%20 == 0):
            test_iou, _ = source.runner.test(
                model=network,
                metric=source.metrics.iou,
                dataloader=test_loader,
                device=device,
                args=args, 
                l2a=label_to_anno, 
                path = results_img_dir, 
                class_obj=class_obj, 
                logs = logs, 
                epoch = epoch, 
                )
            print("test iou: ", test_iou)
            print("test miou: ", np.mean(test_iou))
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

        l = logs_valid[criterion.name]
        m = np.mean(logs_valid[metric.name])
        if best_loss > l:
            best_loss = l
            #torch.save(network, os.path.join(outdir, f"{network_fout}.pth"))
            torch.save(fcmixnet.state_dict(), os.path.join(outdir, f"fcmixnet.pth"))
            torch.save(network.state_dict(), os.path.join(outdir, f"{network_fout}.pth"))
            print("Model saved based on loss !")
        if best_metrix < m:
            best_metrix = m
            torch.save(fcmixnet.state_dict(), os.path.join(outdir, f"fcmixnet_metrix.pth"))
            torch.save(network.state_dict(), os.path.join(outdir, f"{network_fout}_metrix.pth"))
            print("Model saved based on matrix !")


    print(f"Completed: {(time.time() - start)/60.0:.4f} min.")
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
    #network = torch.load(learned_path)
    network.load_state_dict(torch.load(learned_path))

    testset = source.dataset.Dataset(img_file[2], ano_file[2], ref_list = [], \
        classes=classes, train=False, final = True, args=args)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)


    test_iou, df = source.runner.final(
            model=network,
            metric=source.metrics.iou,
            dataloader=test_loader,
            device=device,
            args=args, 
            l2a=label_to_anno, 
            path = outdir, 
            class_obj=class_obj, 
            logs = logs, 
            epoch = -1, 
        )
    print(test_iou)
    print(np.mean(test_iou))
    df.to_csv(outdir +"/log_output.csv")
    print("save test results")

if __name__ == '__main__':
    base = source.options.BaseOptions()
    args = base.initialize()
    base.show_options(args)
    main(args)
    