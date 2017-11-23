import numpy as np


def get_hinge_loss(X, idea_index, W):
    scores = W.dot(X)
    local_loss = np.maximum(0, scores - scores[idea_index] + 1)
    local_loss[idea_index] = 0
    global_loss = np.sum(local_loss)
    return global_loss
