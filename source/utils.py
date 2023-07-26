import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pathlib


def progress(train_logs, valid_logs, loss_nm, metric_nm, nepochs, outdir, fn_out, test_ious, opt):
    loss_t = [dic[loss_nm] for dic in train_logs]
    if opt.randomize and opt.optimize:
        mask_loss_t = [dic["mask"] for dic in train_logs]
    loss_v = [dic[loss_nm] for dic in valid_logs]
    score_t = [np.mean(dic[metric_nm]) for dic in train_logs]
    score_v = [np.mean(dic[metric_nm]) for dic in valid_logs]

    epochs = range(0, len(score_t))
    plt.figure(figsize=(12, 5))

    # Train and validation metric
    # ---------------------------
    plt.subplot(1, 2, 1)

    idx = np.nonzero(score_t == max(score_t))[0][0]
    label = f"Train, {metric_nm}={max(score_t):6.4f} in Epoch={idx}"
    plt.plot(epochs, score_t, "b", label=label)

    if len(score_v)>0:
        idx = np.nonzero(score_v == max(score_v))[0][0] + (len(score_t) - len(score_v))
        label = f"Valid, {metric_nm}={max(score_v):6.4f} in Epoch={idx}"
        plt.plot(range(len(score_t) - len(score_v), len(score_t)), score_v, "r", label=label)

    if len(test_ious)>0:
        plt.scatter(test_ious.keys(), test_ious.values(), c = "deepskyblue", s = 60, label = "Test")


    plt.title("Training and Validation Metric")
    plt.xlabel("Epochs")
    plt.xlim(0, nepochs)
    plt.ylabel(metric_nm)
    plt.ylim(0, 1)
    plt.legend()


    # Train and validation loss
    # -------------------------
    plt.subplot(1, 2, 2)
    ymax = max(max(loss_t), max(loss_v))
    ymin = min(min(loss_t), min(loss_v))
    ymax = 1 if ymax <= 1 else ymax + 0.5
    ymin = 0 if ymin <= 0.5 else ymin - 2.0

    idx = np.nonzero(loss_t == min(loss_t))[0][0]
    label = f"Train {loss_nm}={min(loss_t):6.4f} in Epoch:{idx}"
    plt.plot(epochs, loss_t, "b", label=label)
    if opt.randomize and opt.optimize:
        plt.plot(epochs, mask_loss_t, "b", linestyle="dashed", label="loss for masks")

    idx = np.nonzero(loss_v == min(loss_v))[0][0]
    label = f"Valid {loss_nm}={min(loss_v):6.4f} in Epoch:{idx}"
    plt.plot(epochs, loss_v, "r", label=label)

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.xlim(0, nepochs)
    plt.ylabel("Loss")
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig(f"{outdir}/{fn_out}.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    return

def save_predicted_img(sample, gt, pr, classes, label_to_anno, path):
    rgb_fname = pathlib.Path(sample["fn"][0]).stem

    out_gt = np.zeros(shape=pr.shape+ (3,), dtype="uint8")
    out_pr = np.zeros(shape=pr.shape+ (3,), dtype="uint8")
    raw = np.array(Image.open(sample["fn"][0]))

    for c in [0] + classes:
        out_gt[gt == c, :] = label_to_anno[c]
        out_pr[pr == c, :] = label_to_anno[c]
    
    plt.figure(figsize=(15,5))
    plt.subplot(1, 3, 1)
    plt.imshow(out_gt)
    plt.title("gt")
    plt.subplot(1, 3, 2)
    plt.imshow(out_pr)
    plt.title("predicted")
    plt.subplot(1, 3, 3)
    plt.imshow(raw)
    plt.title("image")
    plt.savefig(path + f"/{rgb_fname}.png")
    #plt.show()
    plt.clf()
    plt.close()

    return rgb_fname

def show_test_result(matrix, class_obj_list, path):
    
    # heatmap
    sns.heatmap(matrix*100, annot=True, cmap = "Blues", \
        xticklabels=class_obj_list, yticklabels=class_obj_list, fmt=".2f")
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.savefig(f"{path}/test_matrix.png", bbox_inches='tight')
    plt.show()
    return None
