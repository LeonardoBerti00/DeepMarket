import numpy as np
from scipy.stats import entropy
import scipy.stats as stats

######################### Jensen-Shannon Divergence #########################
# It is used to measure the similarity between two probability distributions ---> 0: same distribution, 1: different distribution ---> [0,1] range #

class JSDCalculator:
    def __init__(self, dataset_1, dataset_2):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2

    def jensen_shannon_divergence(self, p, q):
        p = np.asarray(p)
        q = np.asarray(q)
        p = p / p.sum()
        q = q / q.sum()
        m = (p + q) / 2
        kl_p = stats.entropy(p, m)
        kl_q = stats.entropy(q, m)
        jsd = (kl_p + kl_q) / 2
        return jsd

    def calculate_jsd(self):
        jsd_features = []
        for i in range(self.dataset_1.shape[1]):
            feature_1 = self.dataset_1[:, i]
            feature_2 = self.dataset_2[:, i]
            jsd = self.jensen_shannon_divergence(feature_1, feature_2)
            jsd_features.append(jsd)
        return jsd_features

######################### Kolmogorov-Smironov statistic (test) #########################
# It is used to measure the similarity between two probability distributions #

class KSCalculator:
    def __init__(self, historical_data, generated_data):
        self.p_1 = historical_data
        self.q_1 = generated_data

    def calculate_ks(self):
        ks_stat, p_value = stats.ks_2samp(np.mean(self.p_1, axis=0), np.mean(self.q_1,axis=0))
        print(f"Test statistic is {ks_stat:.4f}")
        print(f"Test value is{p_value:.4f}")
        if p_value < 0.05:
            print("We reject the null hypothesis that the two datasets come from the same distribution")
        else:
            print("We do not reject the null hypothesis that the two datasets come from the same distribution")
        return ks_stat, p_value
    
    def calculate_ks_features(self):
        ks_results = []
        for i in range(self.p_1.shape[1]):
            feature_1 = self.p_1[:, i]
            feature_2 = self.q_1[:, i]
            ks_stat, p_value = stats.ks_2samp(feature_1, feature_2)
            ks_results.append((ks_stat, p_value))
        return ks_results



'''
JSD is a symmetric and smoothed version of Kullback-Leibler Divergence (KLD) that measures the similarity between two probability distributions. It is defined as the square root of the Jensen-Shannon Divergence,
which is a smoothed version of KLD. JSD is commonly used in probability theory and information theory to compare probability distributions.

KS statistic, on the other hand, is a non-parametric test that compares two probability distributions by measuring the maximum difference between their cumulative distribution functions (CDFs).

It is based on the empirical distribution functions of the two datasets and is used to test whether two samples are drawn from the same distribution.
'''
