import numpy as np
from scipy.stats import entropy
import scipy.stats as stats

######################### Jensen-Shannon Divergence #########################
# It is used to measure the similarity between two probability distributions ---> 0: same distribution, 1: different distribution #


def JSD(q, p):
    jsd_list = []
    if (type(q) != np.ndarray):
        q = np.array(q)
    if (type(p) != np.ndarray):
        p = np.array(p)
    for i in range(p.shape[1]):
        p_i = p[:, i]
        p_i /= np.sum(p_i)
        q_i = q[:, i]
        q_i /= np.sum(q_i)
        jsd_list.append(np.sqrt(0.5 * (entropy(p_i, q_i) + entropy(q_i, p_i))))
    return jsd_list



######################### Kolmogorov-Smironov statistic (test) #########################
# It is used to measure the similarity between two probability distributions #

class KSCalculator:
    def __init__(self, historical_data, generated_data):
        self.p_1 = np.mean(historical_data, axis=0)
        self.q_1 = np.mean(generated_data, axis=0)

    def calculate_ks(self):
        ks_stat, p_value = stats.ks_2samp(self.p_1, self.q_1)
        print(f"Test statistic is {ks_stat:.4f}")
        print(f"Test value si{p_value:.4f}")
        if p_value < 0.05:
            print("We reject the null hypothesis that the two datasets come from the same distribution")
        else:
            print("We do not reject the null hypothesis that the two datasets come from the same distribution")
        return ks_stat, p_value
    


'''
JSD is a symmetric and smoothed version of Kullback-Leibler Divergence (KLD) that measures the similarity between two probability distributions. It is defined as the square root of the Jensen-Shannon Divergence,
which is a smoothed version of KLD. JSD is commonly used in probability theory and information theory to compare probability distributions.

KS statistic, on the other hand, is a non-parametric test that compares two probability distributions by measuring the maximum difference between their cumulative distribution functions (CDFs).

It is based on the empirical distribution functions of the two datasets and is used to test whether two samples are drawn from the same distribution.
'''
