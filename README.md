# sketching-with-chebyshev
meow

in run uci modify solve regression as follows:

def solve_reg(x_tr, y_tr, x_te, y_te, lam, y_std=1.0):
    n, d = x_tr.shape

     # Ensure all inputs are torch.DoubleTensor regardless of source
    x_tr = x_tr.double() if isinstance(x_tr, torch.Tensor) else torch.from_numpy(x_tr).double()
    y_tr = y_tr.double() if isinstance(y_tr, torch.Tensor) else torch.from_numpy(y_tr).double()
    x_te = x_te.double() if isinstance(x_te, torch.Tensor) else torch.from_numpy(x_te).double()
    y_te = y_te.double() if isinstance(y_te, torch.Tensor) else torch.from_numpy(y_te).double()

    if x_tr.shape[0] > x_tr.shape[1]:
        b = x_tr.T @ (y_tr[:, None] if y_tr.dim() == 1 else y_tr)
        y_pred = x_te @ torch.linalg.solve(x_tr.T @ x_tr + lam * torch.eye(d, dtype=torch.float64), b)
    else:
        b = y_tr[:, None] if y_tr.dim() == 1 else y_tr
        y_pred = x_te @ (x_tr.T @ torch.linalg.solve(x_tr @ x_tr.T + lam * torch.eye(n, dtype=torch.float64), b))

    mse = mean_squared_error((y_pred * y_std).cpu(), (y_te * y_std).cpu())
    acc = ((y_pred.argmax(axis=1) == y_te.argmax(axis=1)) * 1.0).mean().item()
    return mse, acc

    to run it on cifar10
