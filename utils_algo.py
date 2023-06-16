import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment



def cluster_acc(y, y_pred):
    """Unsupervised Clustering Accuracy
    """
    assert len(y_pred) == len(y)
    u = np.unique(np.concatenate((y, y_pred)))
    n_clusters = len(u)
    mapping = dict(zip(u, range(n_clusters)))
    reward_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for y_pred_, y_ in zip(y_pred, y):
        if y_ in mapping:
            reward_matrix[mapping[y_pred_], mapping[y_]] += 1
    cost_matrix = reward_matrix.max() - reward_matrix
    row_assign, col_assign = linear_sum_assignment(cost_matrix)

    # Construct optimal assignments matrix
    row_assign = row_assign.reshape((-1, 1))  # (n,) to (n, 1) reshape
    col_assign = col_assign.reshape((-1, 1))  # (n,) to (n, 1) reshape
    assignments = np.concatenate((row_assign, col_assign), axis=1)

    optimal_reward = reward_matrix[row_assign, col_assign].sum() * 1.0
    return optimal_reward / y_pred.size





class BCE_softlabels(nn.Module):
    eps = 1e-9  # Avoid calculating log(0). Use the small value of float16.

    def forward(self, prob1, prob2, simi):

        P = prob1.mul_(prob2)
        P = P.sum(1)
        P = P.cpu()
        simi = simi.cpu()
        neglogP = - (simi * torch.log(P + BCE_softlabels.eps) + (1. - simi) * torch.log(1. - P + BCE_softlabels.eps))
        return neglogP.mean()



def PairEnum(x):
    """ Enumerate all pairs of feature in x."""
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    return x1, x2
