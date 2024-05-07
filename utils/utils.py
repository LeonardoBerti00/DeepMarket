from collections import namedtuple
import math
from matplotlib import pyplot as plt
from numpy import cov, sqrt, trace
import numpy
import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
import torch
from tqdm import trange
import constants as cst


#noise scheduler taken from "Improved Denoising Diffusion Probabilistic Models"
def noise_scheduler(num_diffusion_timesteps, max_beta=0.99):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32).to(cst.DEVICE, non_blocking=True)


#formula taken from "Denoising Diffusion Probabilistic Models"
def compute_mean_tilde_t(x_0, x_T, alpha_cumprod_t, alpha_cumprod_t_1, beta_t, alpha_t):
    # alpha_cumprod_t_1 is alpha_cumprod(t-1)
    return torch.sqrt(alpha_cumprod_t_1)*beta_t*x_0 / (1-alpha_cumprod_t) + torch.sqrt(alpha_t)*(1-alpha_cumprod_t_1)*x_T / (1-alpha_cumprod_t)


def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):

    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings.to(cst.DEVICE, non_blocking=True)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size)
    torch.nn.init.kaiming_normal_(layer.weight)
    return layer

# calculate frechet inception distance
def calculate_ftsd(h1, h2):
    # calculate mean and covariance statistics
    mu1, sigma1 = h1.mean(axis=0), cov(h1, rowvar=False)
    mu2, sigma2 = h2.mean(axis=0), cov(h2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrt(sigma1.dot(sigma2))
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# the next functions are taken from https://github.com/msmsajjadi/precision-recall-distributions/blob/master/prd_score.py#L108
def compute_prd_from_embedding(eval_data, ref_data, num_clusters=20,
                               num_angles=1001, num_runs=10,
                               enforce_balance=True):
  """Computes PRD data from sample embeddings.

  The points from both distributions are mixed and then clustered. This leads
  to a pair of histograms of discrete distributions over the cluster centers
  on which the PRD algorithm is executed.

  The number of points in eval_data and ref_data must be equal since
  unbalanced distributions bias the clustering towards the larger dataset. The
  check can be disabled by setting the enforce_balance flag to False (not
  recommended).

  Args:
    eval_data: NumPy array of data points from the distribution to be evaluated.
    ref_data: NumPy array of data points from the reference distribution.
    num_clusters: Number of cluster centers to fit. The default value is 20.
    num_angles: Number of angles for which to compute PRD. Must be in [3, 1e6].
                The default value is 1001.
    num_runs: Number of independent runs over which to average the PRD data.
    enforce_balance: If enabled, throws exception if eval_data and ref_data do
                     not have the same length. The default value is True.

  Returns:
    precision: NumPy array of shape [num_angles] with the precision for the
               different ratios.
    recall: NumPy array of shape [num_angles] with the recall for the different
            ratios.

  Raises:
    ValueError: If len(eval_data) != len(ref_data) and enforce_balance is set to
                True.
  """

  if enforce_balance and len(eval_data) != len(ref_data):
    raise ValueError(
        'The number of points in eval_data %d is not equal to the number of '
        'points in ref_data %d. To disable this exception, set enforce_balance '
        'to False (not recommended).' % (len(eval_data), len(ref_data)))

  eval_data = np.array(eval_data, dtype=np.float64)
  ref_data = np.array(ref_data, dtype=np.float64)
  precisions = []
  recalls = []
  for _ in range(num_runs):
    eval_dist, ref_dist = _cluster_into_bins(eval_data, ref_data, num_clusters)
    precision, recall = compute_prd(eval_dist, ref_dist, num_angles)
    precisions.append(precision)
    recalls.append(recall)
  precision = np.mean(precisions, axis=0)
  recall = np.mean(recalls, axis=0)
  result = np.concatenate([[precision], [recall]]).T
  return result

def _cluster_into_bins(eval_data, ref_data, num_clusters):
  """Clusters the union of the data points and returns the cluster distribution.

  Clusters the union of eval_data and ref_data into num_clusters using minibatch
  k-means. Then, for each cluster, it computes the number of points from
  eval_data and ref_data.

  Args:
    eval_data: NumPy array of data points from the distribution to be evaluated.
    ref_data: NumPy array of data points from the reference distribution.
    num_clusters: Number of cluster centers to fit.

  Returns:
    Two NumPy arrays, each of size num_clusters, where i-th entry represents the
    number of points assigned to the i-th cluster.
  """

  cluster_data = np.vstack([eval_data, ref_data])
  kmeans = MiniBatchKMeans(n_clusters=num_clusters, n_init=10)
  labels = kmeans.fit(cluster_data).labels_

  eval_labels = labels[:len(eval_data)]
  ref_labels = labels[len(eval_data):]

  eval_bins = np.histogram(eval_labels, bins=num_clusters,
                           range=[0, num_clusters], density=True)[0]
  ref_bins = np.histogram(ref_labels, bins=num_clusters,
                          range=[0, num_clusters], density=True)[0]
  return eval_bins, ref_bins

def compute_prd(eval_dist, ref_dist, num_angles=1001, epsilon=1e-10):
  """Computes the PRD curve for discrete distributions.

  This function computes the PRD curve for the discrete distribution eval_dist
  with respect to the reference distribution ref_dist. This implements the
  algorithm in [arxiv.org/abs/1806.2281349]. The PRD will be computed for an
  equiangular grid of num_angles values between [0, pi/2].

  Args:
    eval_dist: 1D NumPy array or list of floats with the probabilities of the
               different states under the distribution to be evaluated.
    ref_dist: 1D NumPy array or list of floats with the probabilities of the
              different states under the reference distribution.
    num_angles: Number of angles for which to compute PRD. Must be in [3, 1e6].
                The default value is 1001.
    epsilon: Angle for PRD computation in the edge cases 0 and pi/2. The PRD
             will be computes for epsilon and pi/2-epsilon, respectively.
             The default value is 1e-10.

  Returns:
    precision: NumPy array of shape [num_angles] with the precision for the
               different ratios.
    recall: NumPy array of shape [num_angles] with the recall for the different
            ratios.

  Raises:
    ValueError: If not 0 < epsilon <= 0.1.
    ValueError: If num_angles < 3.
  """

  if not (epsilon > 0 and epsilon < 0.1):
    raise ValueError('epsilon must be in (0, 0.1] but is %s.' % str(epsilon))
  if not (num_angles >= 3 and num_angles <= 1e6):
    raise ValueError('num_angles must be in [3, 1e6] but is %d.' % num_angles)

  # Compute slopes for linearly spaced angles between [0, pi/2]
  angles = np.linspace(epsilon, np.pi/2 - epsilon, num=num_angles)
  slopes = np.tan(angles)

  # Broadcast slopes so that second dimension will be states of the distribution
  slopes_2d = np.expand_dims(slopes, 1)

  # Broadcast distributions so that first dimension represents the angles
  ref_dist_2d = np.expand_dims(ref_dist, 0)
  eval_dist_2d = np.expand_dims(eval_dist, 0)

  # Compute precision and recall for all angles in one step via broadcasting
  precision = np.minimum(ref_dist_2d*slopes_2d, eval_dist_2d).sum(axis=1)
  recall = precision / slopes

  # handle numerical instabilities leaing to precision/recall just above 1
  max_val = max(np.max(precision), np.max(recall))
  if max_val > 1.001:
    raise ValueError('Detected value > 1.001, this should not happen.')
  precision = np.clip(precision, 0, 1)
  recall = np.clip(recall, 0, 1)

  return precision, recall

def plot(precision_recall_pairs, labels=None, out_path=None,
         legend_loc='lower left', dpi=300):
  """Plots precision recall curves for distributions.

  Creates the PRD plot for the given data and stores the plot in a given path.

  Args:
    precision_recall_pairs: List of prd_data to plot. Each item in this list is
                            a 2D array of precision and recall values for the
                            same number of ratios.
    labels: Optional list of labels of same length as list_of_prd_data. The
            default value is None.
    out_path: Output path for the resulting plot. If None, the plot will be
              opened via plt.show(). The default value is None.
    legend_loc: Location of the legend. The default value is 'lower left'.
    dpi: Dots per inch (DPI) for the figure. The default value is 150.

  Raises:
    ValueError: If labels is a list of different length than list_of_prd_data.
  """

  if labels is not None and len(labels) != len(precision_recall_pairs):
    raise ValueError(
        'Length of labels %d must be identical to length of '
        'precision_recall_pairs %d.'
        % (len(labels), len(precision_recall_pairs)))

  fig = plt.figure(figsize=(3.5, 3.5), dpi=dpi)
  plot_handle = fig.add_subplot(111)
  plot_handle.tick_params(axis='both', which='major', labelsize=12)

  for i in range(len(precision_recall_pairs)):
    precision, recall = precision_recall_pairs[i]
    label = labels[i] if labels is not None else None
    plt.plot(recall, precision, label=label, alpha=0.5, linewidth=3)

  if labels is not None:
    plt.legend(loc=legend_loc)

  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.xlabel('Recall', fontsize=12)
  plt.ylabel('Precision', fontsize=12)
  plt.tight_layout()
  if out_path is None:
    plt.show()
  else:
    plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    

# the next class and 4 function are ported from https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/master/improved_precision_recall.py#L239

import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

Manifold = namedtuple('Manifold', ['features', 'radii'])
PrecisionAndRecall = namedtuple('PrecisinoAndRecall', ['precision', 'recall'])


class IPR():
    def __init__(self, batch_size=50, k=3, num_samples=10000):
        self.manifold_ref = None
        self.batch_size = batch_size
        self.k = k
        self.num_samples = num_samples

    def __call__(self, subject):
        return self.precision_and_recall(subject)

    def precision_and_recall(self, subject):
        '''
        Compute precision and recall for given subject
        reference should be precomputed by IPR.compute_manifold_ref()
        args:
            subject: path or images
                path: a directory containing images or precalculated .npz file
                images: torch.Tensor of N x C x H x W
        returns:
            PrecisionAndRecall
        '''
        assert self.manifold_ref is not None, "call IPR.compute_manifold_ref() first"

        manifold_subject = self.compute_manifold(subject)
        precision = compute_metric(self.manifold_ref, manifold_subject.features, 'computing precision...')
        recall = compute_metric(manifold_subject, self.manifold_ref.features, 'computing recall...')
        return PrecisionAndRecall(precision, recall)

    def compute_manifold_ref(self, path):
        self.manifold_ref = self.compute_manifold(path)

    def compute_manifold(self, feats):
        distances = compute_pairwise_distances(feats)
        radii = distances2radii(distances, k=self.k)
        return Manifold(feats, radii)

def compute_pairwise_distances(X, Y=None):
    '''
    args:
        X: np.array of shape N x dim
        Y: np.array of shape N x dim
    returns:
        N x N symmetric np.array
    '''
    num_X = X.shape[0]
    if Y is None:
        num_Y = num_X
    else:
        num_Y = Y.shape[0]
    X = X.astype(np.float64)  # to prevent underflow
    X_norm_square = np.sum(X**2, axis=1, keepdims=True)
    if Y is None:
        Y_norm_square = X_norm_square
    else:
        Y_norm_square = np.sum(Y**2, axis=1, keepdims=True)
    X_square = np.repeat(X_norm_square, num_Y, axis=1)
    Y_square = np.repeat(Y_norm_square.T, num_X, axis=0)
    if Y is None:
        Y = X
    XY = np.dot(X, Y.T)
    diff_square = X_square - 2*XY + Y_square

    # check negative distance
    min_diff_square = diff_square.min()
    if min_diff_square < 0:
        idx = diff_square < 0
        diff_square[idx] = 0
        print('WARNING: %d negative diff_squares found and set to zero, min_diff_square=' % idx.sum(),
              min_diff_square)

    distances = np.sqrt(diff_square)
    return distances

def distances2radii(distances, k=3):
    num_features = distances.shape[0]
    radii = np.zeros(num_features)
    for i in range(num_features):
        radii[i] = get_kth_value(distances[i], k=k)
    return radii

def compute_metric(manifold_ref, feats_subject, desc=''):
    num_subjects = feats_subject.shape[0]
    count = 0
    dist = compute_pairwise_distances(manifold_ref.features, feats_subject)
    for i in trange(num_subjects, desc=desc):
        count += (dist[:, i] < manifold_ref.radii).any()
    return count / num_subjects

def get_kth_value(np_array, k):
    kprime = k+1  # kth NN should be (k+1)th because closest one is itself
    idx = np.argpartition(np_array, kprime)
    k_smallests = np_array[idx[:kprime]]
    kth_value = k_smallests.max()
    return kth_value


# the next functions are taken from https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
import sklearn.metrics

def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)