import torch
import torch.distributions as td
import numpy as np

def calculate_new_exposure(A_t,weights,c,gamma,ab_1_peak,ab_2_peak):
    """
    inputs:
    A_t: The exposure at time t (bound by -1 and 1) as torch.(Float)Tensor
    weights: List of weights for mixture model in order [dark_peak_prob,bright_peak_prob,rebound_prob]
    c: Sum of alpha and beta for rebound distribution
    gamma: The rebound effect is modeled with an exponential model, bound from 0 (immediately jump back to zero) to 1 (never jump back)
    ab_1_peak: alpha for dark peak/ beta for bright peak
    ab_2_peak: beta for dark peak/ alpha for bright peak
    """
    alpha_tp1 = (gamma*A_t+1)/2*c
    beta_tp1 = c-alpha_tp1
    cat = td.Categorical(torch.FloatTensor(weights))
    b = td.Beta(torch.FloatTensor([ab_1_peak,ab_2_peak,alpha_tp1.item()]),torch.FloatTensor([ab_2_peak,ab_1_peak,beta_tp1.item()]))
    m = td.MixtureSameFamily(cat,b)
    #OUTPUT: Mean and std
    # return m.sample(torch.Size([1]))*2-1
    return m.mean.item(), m.stddev.item()


def exposure_mean_std(exposure):
    weights = [0.005,0.005,0.99]
    c = 50000
    gamma = 0.7
    ab_1_peak = 2
    ab_2_peak = 55
    if isinstance(exposure, float):
        exposure = torch.Tensor([exposure])
    return calculate_new_exposure(exposure, weights, c, gamma, ab_1_peak, ab_2_peak)


if __name__=="__main__":
    A_0 = torch.FloatTensor([0])
    weights = [0.005,0.005,0.99]
    c = 50000
    gamma = 0.7
    ab_1_peak = 2
    ab_2_peak = 55
    t_steps = 200
    A = [A_0]

    for t in range(t_steps):
        A.append(calculate_new_exposure(A[-1],weights,c,gamma,ab_1_peak,ab_2_peak))
    
    A = [a.item() for a in A]

# fig = figure.Figure()
# ax = axes.CartesianAxis()
# ax.title = "Exposure Correction Trajectory"
# ax.xlabel = r"$t$"
# ax.ylabel = r"A"
# ax.xmin, ax.xmax = 0, 200
# ax.ymin, ax.ymax = -1, 1
# p = plots.Plot(np.arange(201),np.array(A))
# fig.axis = ax
# fig.plots = [p]
# fig.write("./figs/trajectory_1.pdf")