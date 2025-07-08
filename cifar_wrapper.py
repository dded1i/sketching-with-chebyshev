import random
from tqdm import tqdm
import numpy as np
import torch
import argparse

from datasets import get_cifar10
from cntk_sketch import OblvFeatCNTK, CNTKSketch
from run_uci_regression import solve_reg
import time


def _one_hot(x, k, dtype=np.float32):
    return np.array(x[:, None] == np.arange(k), dtype)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def hyperparams_search_acc(LAMBDA_LIST, X, y, train_fold, valid_fold):
    best_acc = -np.inf
    best_lam = np.inf

    for lam in LAMBDA_LIST:
        _, acc = solve_reg(X[train_fold], y[train_fold], X[valid_fold], y[valid_fold], lam)
        if acc >= best_acc:
            best_lam = lam
            best_acc = acc

    return {"best_lam": best_lam, "best_acc": best_acc}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--method', default='cntksketch', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--feat_dim', default=128, type=int)
    parser.add_argument('--num_cv_layers', default=3, type=int)
    parser.add_argument('--ns_deg', default=2, type=int)
    parser.add_argument('--filter_size', default=4, type=int)
    args = parser.parse_args()

    for name_, val_ in args.__dict__.items():
        print(f"{name_:>15} : {val_}")

    set_random_seed(args.seed)

    x_tr, y_tr, x_te, y_te = get_cifar10()
    x_tr = x_tr[:32]; y_tr = y_tr[:32]
    x_te = x_te[:8];  y_te = y_te[:8]

    num_classes = 10
    y_tr = _one_hot(y_tr, 10)
    y_te = _one_hot(y_te, 10)
    y_tr -= (1.0 / num_classes)
    y_te -= (1.0 / num_classes)

    x_tr = torch.FloatTensor(x_tr)
    y_tr = torch.FloatTensor(y_tr)
    x_te = torch.FloatTensor(x_te)
    y_te = torch.FloatTensor(y_te)

    num_channels = x_tr.shape[-1]
    x = torch.cat((x_tr, x_te), axis=0).moveaxis(3, 1)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device : {dev}")

    cntk_sketch = CNTKSketch(args.filter_size, num_channels, args.feat_dim, args.ns_deg, args.num_cv_layers, dev)


    try:
        with torch.no_grad():
            x_batch = x.to(dev)
            start_time = time.time()
            z = OblvFeatCNTK(cntk_sketch, x_batch)
            elapsed = time.time() - start_time
            print(f"Sketching time: {elapsed:.3f} seconds")
            if torch.cuda.is_available():
                z = z.detach().cpu()
    except Exception as e:
        print(e)
        print(f"[ERROR] CNTK sketching failed!")
        return

    z_tr = z[:len(x_tr)]
    z_te = z[len(x_tr):]

    set_random_seed(args.seed)

    rand_idx = np.random.permutation(len(x_tr))
    train_fold = torch.tensor(rand_idx[:24])
    valid_fold = torch.tensor(rand_idx[24:32])
    LAMBDA_LIST = [2**i for i in range(-6, 6)]

    res = hyperparams_search_acc(LAMBDA_LIST, z_tr, y_tr, train_fold, valid_fold)
    best_lam = res['best_lam']

    mse, acc = solve_reg(z_tr, y_tr, z_te, y_te, best_lam)
    print(f"mse: {mse}, acc: {acc}")

if __name__ == "__main__":
    print("[DEBUG] main() started")

    main()
