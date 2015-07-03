Three Entropy Estimators
========================

Three estimators for the entropy of continuous random variables (one dimension)

1. `entropy_bin()` uses the common histogram approach. In addition to the data, it
requires specifying a bin width.
2. `entropy_ci()` uses the first order correlation integral, which is like a niave
kernel density estimator. In addition to the data, it required specifying a
neighborhood radius (kernel bandwidth), which analogous to a (half) bin width
for the histogram estimator.
3. `entropy_nn()` uses the distribution of nearest neighbor distances. It requires no
adjustable parameters.

Using some simulations with various bandwidths, my experience is that the
nearest neighbor estimator has the lowest bias, but the highest variance. The
correlation integral estimator is probably the best, especially with a well
chosen neighbor radius. The histogram methods tends to underestimate the entropy.
I suspect a kernel density estimator using a gaussian kernel would be even better,
but that is not implemented.
