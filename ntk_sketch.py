import numpy as np
import torch
import torch.fft
from torch import linalg as LA
import math
import quadprog
import time

#relates to step 3 of seudo
def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    """
     Solve the quadratic programming problem of the form
     min 1/2x^TPx + q^Tx s.t.
     Gx<=h, Ax=b

     finds the best-fit polynomial coefficients c_j

     :param P: positive semi-definite matrix
     :param q:  vector for the linear term
     :param G: inequality constraint
     :param h: inequality constraint
     :param A: equality constraint
     :param b: same
     :return: optimal weights for polynomial basis
     """
    # make sure P is symmetric
    #all the code is basic parametrization to solve the problem through QP solver
    qp_G = .5 * (P + P.T + 0.00001 * np.eye(P.shape[0]))
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def get_poly_approx_ntk(num_layers, degree):
    """
    Computes a polynomial approximation to the NTK recursion function using Chebyshev-like interpolation and QP.
    Find a d-degree polynomial that approximates the ReLU NTK kernel function
    :param num_layers:
    :param degree: polynomial degree
    :return:
    """
    # number of extra interpolation points (to imporve approx?)
    n = 15 * num_layers + 5 * degree
    #Initializes matrix Y to store computed NTK kernels across layers
    Y = np.zeros((201 + n, num_layers + 1))

    #  Build interpolation points (uniform + Chebyshev) for stable polynomial fit
    x_linear = np.linspace(-1.0, 1.0, num=201)
    x_cosine = np.cos((2 * np.arange(n) + 1) * np.pi / (4 * n))
    Y[:, 0] = np.sort(np.concatenate((x_linear, x_cosine), axis=0))
    #alpha
    #number of total evaluation points (for downstream matrix dimensions?)
    m_ = Y.shape[0]

    # recursively computes the NTK at each alpha
    for i in range(num_layers):
        Y[:, i + 1] = (np.sqrt(1 - Y[:, i]**2) + Y[:, i] * (np.pi - np.arccos(Y[:, i]))) / np.pi

    # Compute full NTK kernel
    y = np.zeros(m_)
    for i in range(num_layers + 1):
        z = Y[:, i]
        for j in range(i, num_layers):
            z = z * (np.pi - np.arccos(Y[:, j])) / np.pi
        y = y + z

    #Create polynomial basis (Vandermonde) matrix Z_{ij} -- basis for fitting y
    Z = np.zeros((m_, degree + 1))
    Z[:, 0] = np.ones(m_)
    for i in range(degree):
        Z[:, i + 1] = Z[:, i] * Y[:, 0]


    #Solve weighted least squares + constraints
    #Weights over interpolation points
    weight_ = np.linspace(0.0, 1.0, num=m_) + 2 / num_layers

    #Apply weights to target and basis
    w = y * weight_
    U = Z.T * weight_

    # Solve QP to find optimal coefficients
    coeff = quadprog_solve_qp(np.dot(U, U.T), -np.dot(U, w),
                              np.concatenate((Z[0:m_ - 1, :] - Z[1:m_, :], -np.eye(degree + 1)), axis=0),
                              np.zeros(degree + m_))
    #coeff[coeff < 0.001] = 0  #Run csetting, stricter threshold
    coeff = coeff / coeff.sum()  # Normalize so all c_j sum to 1

    return coeff


def TSRHTCmplx(X1, X2, P, D):
    """
    Computes randomized Hadamard-like projections using FFTs
    dimension is reduced by random sign flipping (D) and index selection (P)
    :param X1:
    :param X2:
    :param P:
    :param D:
    :return:
    """
    Xhat1 = torch.fft.fftn(X1 * D[0, :], dim=1)[:, P[0, :]]
    Xhat2 = torch.fft.fftn(X2 * D[1, :], dim=1)[:, P[1, :]]

    Y = np.sqrt(1 / P.shape[1]) * (Xhat1 * Xhat2)

    return Y


class TensorSketch:

    def __init__(self, d, m, q, dev):
        self.d = d
        self.m = m
        self.q = q
        self.device_ = dev

        self.Tree_D = [0 for i in range((self.q - 1).bit_length())]
        self.Tree_P = [0 for i in range((self.q - 1).bit_length())]

        m_ = int(self.m / 4)
        q_ = int(self.q / 2)
        #initializes sketch tree:
        #At level 0 sketch input dim d, at deeper levels -- previous sketch level

        for i in range((self.q - 1).bit_length()):
            if i == 0:
                self.Tree_P[i] = torch.from_numpy(np.random.choice(self.d, (q_, 2, m_))).to(self.device_)
                self.Tree_D[i] = torch.from_numpy(np.random.choice((-1, 1), (q_, 2, self.d))).to(self.device_)
            else:
                self.Tree_P[i] = torch.from_numpy(np.random.choice(m_, (q_, 2, m_))).to(self.device_)
                self.Tree_D[i] = torch.from_numpy(np.random.choice((-1, 1), (q_, 2, m_))).to(self.device_)
            q_ = int(q_ / 2)

        self.D = torch.from_numpy(np.random.choice((-1, 1), self.q * m_)).to(self.device_)
        self.P = torch.from_numpy(np.random.choice(self.q * m_, int(self.m / 2 - 1))).to(self.device_)

    def Sketch(self, X):
        n = X.shape[0]
        lgq = len(self.Tree_D)
        V = [0 for i in range(lgq)]
        E1 = torch.cat((torch.ones((n, 1), device=self.device_), torch.zeros((n, X.shape[1] - 1), device=self.device_)),
                       1)
        #feature maps constructiob
        for i in range(lgq):
            q = self.Tree_D[i].shape[0]
            V[i] = torch.zeros((q, n, self.Tree_P[i].shape[2]), dtype=torch.cfloat, device=self.device_)
            for j in range(q):
                if i == 0:
                    V[i][j, :, :] = TSRHTCmplx(X, X, self.Tree_P[i][j, :, :], self.Tree_D[i][j, :, :])
                else:
                    V[i][j, :, :] = TSRHTCmplx(V[i - 1][2 * j, :, :], V[i - 1][2 * j + 1, :, :],
                                               self.Tree_P[i][j, :, :], self.Tree_D[i][j, :, :])
        # "monomial" sketches
        U = [0 for i in range(2**lgq)]
        #sketches x tensor x
        U[0] = V[lgq - 1][0, :, :].detach().clone()
        #Performs recursive sketching to approximate higher tensor powers:
        for j in range(1, len(U)):
            p = int((j - 1) / 2)
            for i in range(lgq):
                if j % (2**(i + 1)) == 0:
                    V[i][p, :, :] = torch.cat((torch.ones((n, 1)), torch.zeros((n, V[i].shape[2] - 1))), 1)
                else:
                    if i == 0:
                        V[i][p, :, :] = TSRHTCmplx(X, E1, self.Tree_P[i][p, :, :], self.Tree_D[i][p, :, :])
                    else:
                        V[i][p, :, :] = TSRHTCmplx(V[i - 1][2 * p, :, :], V[i - 1][2 * p + 1, :, :],
                                                   self.Tree_P[i][p, :, :], self.Tree_D[i][p, :, :])
                p = int(p / 2)
            U[j] = V[lgq - 1][0, :, :].detach().clone()

        return U


def OblvFeat(tensr_sktch, X, coeff):
    q = tensr_sktch.q
    n = X.shape[0]
    norm_X = LA.norm(X, dim=1)
    Normalizer = torch.where(norm_X > 0, norm_X, 1.0)
    Xnormalized = ((X.T / Normalizer).T)
    U = tensr_sktch.Sketch(Xnormalized)
    m = U[0].shape[1]
    #  Weight features with √c_j
    Z = torch.zeros((len(tensr_sktch.D), n), dtype=torch.cfloat, device=tensr_sktch.device_)
    for i in range(q):
        # Z[m*i:m*(i+1)] = np.sqrt(coeff[i+1]) * U[q-i-1].T
        Z[m * i:m * (i + 1)] = coeff[i + 1].sqrt() * U[q - i - 1].T
        U[q - i - 1] = 0

    Z = (np.sqrt(1 / len(tensr_sktch.P)) * torch.fft.fftn(Z.T * tensr_sktch.D, dim=1)[:, tensr_sktch.P])
    Z = (Z.T * Normalizer).T
    return torch.cat((coeff[0].sqrt() * Normalizer.reshape((n, 1)), torch.cat((Z.real, Z.imag), 1)), 1).T
