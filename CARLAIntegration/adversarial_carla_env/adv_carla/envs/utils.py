import numpy as np

def mahalanobis_d(action, mean_disturbance, var_disturbance):
    # Mean action is 0
    action = np.array(action)
    mean = mean_disturbance
    # Assemble the diagonal covariance matrix
    cov = var_disturbance
    big_cov = np.diagflat(cov)

    # subtract the mean from our actions
    dif = np.copy(action)
    dif = dif - mean

    # calculate the Mahalanobis distance
    dist = np.dot(np.dot(dif.T, np.linalg.inv(big_cov)), dif)

    return np.sqrt(dist)


def normalize_observations(obs, mean, std):
    return np.divide(obs - mean, std)
