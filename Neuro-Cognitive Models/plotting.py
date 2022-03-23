import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import seaborn as sns
 
# Codes are taken from bellow
# https://github.com/laurafontanesi/rlssm/blob/main/rlssm/utils.py 
def bci(x, alpha=0.05):
    """Calculate Bayesian credible interval (BCI).
    Parameters
    ----------
    x : array-like
        An array containing MCMC samples.
    alpha : float
        Desired probability of type I error (defaults to 0.05).
    Returns
    -------
    interval : numpy.ndarray
        Array containing the lower and upper bounds of the bci interval.
    """

    interval = np.nanpercentile(x, [(alpha/2)*100, (1-alpha/2)*100])

    return interval

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width.
    Parameters
    ----------
    x : array-like
        An sorted numpy array.
    alpha : float
        Desired probability of type I error (defaults to 0.05).
    Returns
    -------
    hdi_min : float
        The lower bound of the interval.
    hdi_max : float
        The upper bound of the interval.
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max


def hdi(x, alpha=0.05):
    """Calculate highest posterior density (HPD).
        Parameters
        ----------
        x : array-like
            An array containing MCMC samples.
        alpha : float
            Desired probability of type I error (defaults to 0.05).
    Returns
    -------
    interval : numpy.ndarray
        Array containing the lower and upper bounds of the hdi interval.
    """

    # Make a copy of trace
    x = x.copy()
     # Sort univariate node
    sx = np.sort(x)
    interval = np.array(calc_min_interval(sx, alpha))

    return interval

def plot_posterior(data,
                   gridsize=100,
                   clip=None,
                   show_intervals="HDI",
                   alpha_intervals=.05,
                   color='grey',
                   intervals_kws=None,
                   xlabel=None,
                   ylabel=None,
                   title=None,
                   legend=None):
    """Plots a univariate distribution with Bayesian intervals for inference.
    By default, only plots the kernel density estimation using scipy.stats.gaussian_kde.
    Bayesian instervals can be also shown as shaded areas,
    by changing show_intervals to either BCI or HDI.
    Parameters
    ----------
    x : array-like
        Usually samples from a posterior distribution.
    ax : matplotlib.axes.Axes, optional
        If provided, plot on this Axes.
        Default is set to current Axes.
    gridsize : int, default to 100
        Resolution of the kernel density estimation function.
    clip : tuple of (float, float), optional
        Range for the kernel density estimation function.
        Default is min and max values of `x`.
    show_intervals : str, default to "HDI"
        Either "HDI", "BCI", or None.
        HDI is better when the distribution is not simmetrical.
        If None, then no intervals are shown.
    alpha_intervals : float, default to .05
        Alpha level for the intervals calculation.
        Default is 5 percent which gives 95 percent BCIs and HDIs.
    intervals_kws : dict, optional
        Additional arguments for `matplotlib.axes.Axes.fill_between`
        that shows shaded intervals.
        By default, they are 50 percent transparent.
    color : matplotlib.colors
        Color for both the density curve and the intervals.
    Returns
    -------
    ax : matplotlib.axes.Axes
        Returns the `matplotlib.axes.Axes` object with the plot
        for further tweaking.
    """
    fig = plt.figure(figsize=(6,4))
    data = data.reshape(data.shape[0],-1)
    for i in range(data.shape[1]):
        x = data[:,i]
        if clip is None:
            min_x = np.min(x)
            max_x = np.max(x)
        else:
            min_x, max_x = clip

        if intervals_kws is None:
            intervals_kws = {'alpha':.5}

        density = gaussian_kde(x, bw_method='scott')
        xd = np.linspace(min_x, max_x, gridsize)
        yd = density(xd)

        plt.plot(xd, yd, color=color[i])

        if show_intervals is not None:
            if np.sum(show_intervals == np.array(['BCI', 'HDI'])) < 1:
                raise ValueError("must be either None, BCI, or HDI")
            if show_intervals == 'BCI':
                low, high = bci(x, alpha_intervals)
            else:
                low, high = hdi(x, alpha_intervals)
            plt.fill_between(xd[np.logical_and(xd >= low, xd <= high)],
                            yd[np.logical_and(xd >= low, xd <= high)],
                            color=color[i],
                            **intervals_kws) 
   
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if legend is not None:
        fig.legend(labels=legend)
    sns.despine()
