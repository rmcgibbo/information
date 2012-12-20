from __future__ import division
import warnings
import os, sys
import numpy as np
import scipy.spatial
import scipy.weave
import scipy.stats.kde
import matplotlib.pyplot as pp
import bisect

EULER_MASCHERONI = 0.57721566490153286060651209008240243104215933593992

"""
Three estimators for the entropy of continuous random variables (one dimension)

entropy_bin() uses the common histogram approach. In addition to the data, it
requires specifying a bin width.

entropy_ci() uses the first order correlation integral, which is like a niave
kernel density estimator. In addition to the data, it required specifying a
neighborhood radius (kernel bandwidth), which analogous to a (half) bin width
for the histogram estimator.

entropy_nn uses the distribution of nearest neighbor distances. It requires no
adjustable parameters.

Using some simulations with various bandwidths, my experience is that the
nearest neighbor estimator has the lowest bias, but the highest variance. The
correlation integral estimator is probably the best, especially with a well
chosen neighbor radius. The histogram methods tends to underestimate the entropy.

I suspect a kernel density estimator using a gaussian kernel would be even better,
but that is not implemented. The entropy_ci() estimator uses basically a square
kernel.
"""

def entropy_bin(data, width):
    """Entropy of a 1D signal by binning
    
    Data beyond three standard deviations from the mean is discarded
    
    .. [4] Moddemeijer, R. "On estimation of entropy and mutual information of continuous distributions",
           Signal Processing 16 233 (1989)
    """
    #if int(n_bins) != n_bins:
    #    raise ValueError('n_bins must be an int, not %s' % n_bins)
    
    upper = np.mean(data) + 3*np.std(data)
    lower = np.mean(data) - 3*np.std(data)
    bins = np.arange(lower, upper, step=width)

    #bins = np.linspace(lower, upper, n_bins+1)
    bin_widths = bins[1:] - bins[0:-1]
    counts, bins = np.histogram(data, bins)
    p = counts / np.sum(counts)
    
    entropy = -np.sum(np.compress(p != 0, p*(np.log(p) - np.log(bin_widths))))
    
    return entropy
   
def entropy_ke(data):
    """Estimate the entropy of a continuous 1D signal using a kernel approach
    
    
    Ahmad, I., and Lin, P. "A nonparametric estimation of the entropy for
    absolutely continuous distributions (Corresp.)," IEEE Trans. Inf. Theory,
    22 375 (1976)
    """
    pass

def entropy_nn(data, presorted=False):
    """Estimate the entropy of a continuous 1D signal using the distribution
    of nearest neighbor distances

    .. math:: 
    H(x) = \frac{1}{n} \sum_i=1^n \ln(n*\rho_i) + \ln 2 + \gamma
    
    Where `H(x)` is the entropy of the signal x, `n` is the length of the signal, `rho_i`
    is the distance from `x_i` to its nearest neighbor `x_j` in the dataset, and gamma
    is the Euler-Mascheroni constant
    
    Parameters
    ----------
    data : array_like, ndims=1, dtype=float64
        A 1D continuous signal
    presorted : boolean, optional
        Is the `data` array presorted? The rate limiting step of this calculation is sorting
        the data array. So if you've already sorted it, you can make this go a little faster
        by passing true.
    
    Returns
    -------
    h : float
        The estimated entropy

    [1] Beirlant, J. Dudewicz, E. J. Gyoerfi, L. Van der Meulen, E. C.,
        "Nonparametric entropy estimation: An overview", Int. J. Math Stat. Sci.
        6 17 (1997) http://jimbeck.caltech.edu/summerlectures/references/Entropy%20estimation.pdf
    """
    if data.ndim != 1:
        raise ValueError('Only 1D supported')
    data = np.array(data, dtype=np.float64)

    if not presorted:
        data = np.sort(data)

    n = len(data)
    nearest_distances = np.zeros(n, dtype=np.float64)

    # populate the array nearest_distances s.t.
    # nd_i = \min{j < n; j \neq i} (|| data_i - data_j ||) 
    # or in otherwords, nearest_distances[i] gives the distance
    # from data[i] to the other data point which it is nearest to

    # we do this in nlogn time by sorting, but then to the iteration
    # over the sorted array in c because python linear time is way longer
    # than C nlog(n) for moderate n.
    scipy.weave.inline(r'''
    int i;
    double distance, left_distance, right_distance;

    // populate the end points manually
    nearest_distances[0] = data[1] - data[0];
    nearest_distances[n-1] = data[n-1] - data[n-2];

    // iterate over the interior points, checking if they're closer to their
    // left or right neighbor.
    left_distance = nearest_distances[0];
    for (i = 1; i < n - 1; i++) {
        left_distance = right_distance;
        right_distance = data[i + 1] - data[i];
        distance = left_distance < right_distance ? left_distance : right_distance;
        nearest_distances[i] = distance;
    }
    ''', ['data', 'n', 'nearest_distances'])

    return np.mean(np.log(n*nearest_distances)) + np.log(2) + EULER_MASCHERONI

def entropy_ci(data, radius, est_max_neighbors_within_radius=16):
    """Estimate the entropy of a continuous 1D signal using the generalized correlation integral
    
    Parameters
    ----------
    data : array_like

    est_max_neighbors_within_radius : int, optional
         Estimate of the maximum number of datapoints within the specified radius from any trial point.
         # we need to find ALL the data points within radius, but there problem is
         # that kdtree requires we set a number to find (k) in addition to the distance_upper_bound
         # so we need to set the number large enough that we can really find all.                                                                          
         # but since it allocates arrays of size (n x k), we don't want it to be too big.

    Returns
    -------
    h : float
        The estimated entropy
    
    References
    ----------
    [2] Prichard, D. and Theiler, J. "Generalized Redundancies for Time Series Analysis", Physica D. 84 476 (1995)
         http://arxiv.org/pdf/comp-gas/9405006.pdf
    [3] Pawelzik, K. and Schuster, H. G; "Generalized dimensions and entropies from a measured time series",
    Phys. Rev. A; 35 481 (1987)
    """
    n = len(data)
    if data.ndim != 1:
        raise ValueError('Only 1D supported')
    data = np.sort(data)
    
    n_neighbors = np.zeros(n, dtype=np.int)
    for i in xrange(n):
        high = bisect.bisect_left(data, data[i] + radius, lo=i)
        low = bisect.bisect_right(data, data[i] - radius, lo=0, hi=i)
        # number of data points excluding i that are within data[i] - radius and data[i] + radius
        n_neighbors[i] = high - low - 1
        
        # DEBUG
        # assert n_neighbors[i] == np.count_nonzero((data < data[i] + radius) & (data > data[i] - radius)) - 1
        # assert np.all(data[low:high] < data[i] + radius)
        # assert np.all(data[low:high] > data[i] - radius)

    fraction_neighbors = n_neighbors / n
    # exclude the bins where n_neighbors is zero
    # equation 20, 22 in [2]
    
    # note, the paper seems to have left out the log(radius) term, but it's pretty
    # obvious that it's supposed to be there. It's very analogous to the histogram
    # estimator. You will also see an obvious dependence of the mean of the entropy
    # estimate on the bin width if you don't use it
    entropy = -np.mean(np.compress(n_neighbors > 0, np.log(fraction_neighbors) - np.log(2*radius)))
    
    return entropy


def main():
    """Generate some random samples and compare the entropy estimators. This will make some plots.
    
    The idea is to run M trials where we generate N points and calculate their entropy. Then we plot,
    for each estimator, the empirical distribution of the estimates over the M trials.
    """
    
    n_trials = 100
    n_pts = 10000
    bin_entropies_01, nn_entropies, cgi_entropies_01 = [], [], []
    bin_entropies_03, cgi_entropies_03 = [], []
    bin_entropies_02, cgi_entropies_02 = [], []
    
    #entropy_bin2(np.random.randn(1000), 30)
    
    for i in range(n_trials):
        print 'trial', i
        #data = np.random.randn(n_pts)
        data = np.random.exponential(size=n_pts)
        nn_entropies.append(entropy_nn(data))
        bin_entropies_01.append(entropy_bin(data, 0.05))
        cgi_entropies_01.append(entropy_ci(data, 0.05))
        bin_entropies_02.append(entropy_bin(data, 0.2))
        cgi_entropies_02.append(entropy_ci(data, 0.2))
        bin_entropies_03.append(entropy_bin(data, 0.3))
        cgi_entropies_03.append(entropy_ci(data, 0.3))

    pp.figure(figsize=(15,8))
    plot_gkde(nn_entropies, label='nn_entropyes')

    plot_gkde(cgi_entropies_01, label='cgi entropies 0.1')
    plot_gkde(cgi_entropies_02, label='cgi entropies 0.2')
    plot_gkde(cgi_entropies_03, label='cgi entropies 0.3')
    
    plot_gkde(bin_entropies_01, label='bin entropies 0.1')
    plot_gkde(bin_entropies_02, label='bin entropies 0.2')
    plot_gkde(bin_entropies_03, label='bin entropies 0.3')
    
    #analytic = 0.5*np.log(2*np.pi*np.e)
    analytic = 1
    print analytic
    
    pp.plot([analytic, analytic], [0, 20], 'k', linewidth=5)
    pp.legend()
    pp.savefig('fig.png')
    os.system('open fig.png')


def plot_gkde(data, *args, **kwargs):
    """Plot a gaussia kernel density estimator. *args and **kwargs will be passed
    directory to pyplot.plot()"""
    kde = scipy.stats.gaussian_kde(data)
    lower = np.mean(data) - 3*np.std(data)
    upper = np.mean(data) + 3*np.std(data)
    x = np.linspace(lower, upper, 100)
    y = kde(x)

    pp.plot(x, y, *args, **kwargs)
    


if __name__ == '__main__':
    main()
