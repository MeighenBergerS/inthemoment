# -*- coding: utf-8 -*-
# Name: inthemoment.py
# Authors: Stephan Meighen-Berger
# Main interface to the inthemoment package.

# Imports
# Native modules
import numpy as np
from scipy import stats
from scipy.integrate import quad
import scipy.optimize as opt
import numpy.linalg as lin
from time import time
from scipy.linalg import solve
# -----------------------------------------
# Package modules
from .config import config
from .utils import *


class ITM(object):
    """ Interace to the ITM package. This class
    stores all methods required to run the method of moments
    simulations and some comparisons
    Parameters
    ----------
    config : dic
        Configuration dictionary for the simulation

    Returns
    -------
    None
    """
    def __init__(self, userconfig=None):
        # Inputs
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)

        # Create RandomState
        if config["general"]["random state seed"] is None:
            rstate = np.random.RandomState()
        else:
            rstate = np.random.RandomState(
                config["general"]["random state seed"]
            )
        config["runtime"] = {"random state": rstate}
        self._rstate = rstate
        print("Defining the sample")
        if config["pdf"]["name"] == "normal":
            # Setting the underlying pdf
            self._pdf = self._gaussian
        else:
            ValueError("Unknown sample PDF! Check the config file")
        print("Setting the model pdf")
        if config["sample"]["pdf"]["name"] == "normal":
            # Setting the underlying pdf
            self._sample_pdf = self._gaussian
            self._sample_mean = config["sample"]["pdf"]["mean"]
            self._sample_sd = config["sample"]["pdf"]["sd"]
        else:
            ValueError("Unknown PDF! Check the config file")

    def __del__(self):
        """ What to do when the ITM instance is deleted
        """
        print("I am melting.... AHHHHHH!!!!")

    # Generate the sample
    def generate(self):
        """ Generates the sample to work on
        """
        self.sample = self._rstate.normal(
            self._sample_mean, self._sample_sd, config["sample"]["sample size"]
        )
        # Sample to use
        self.subset = (
            self._rstate.choice(self.sample, config["sample"]["subset size"])
        )

    def fit(self) -> np.array:
        """ Runs the method of moments fit and a scipy fit to compare

        Parameters
        ----------
        None

        Returns
        -------
        np.array Shape (4, 2):
            Array containing the resulting fit pairs for mu and sigma
            for the subsets used, the first iteration,
            the corrected on and the scipy fit.
        """
        # ---------------------------------------------------------------------
        print("The subset has the following parameters")
        set_data = self._get_moments(self.subset)
        mu_subset = set_data[0]
        sig_subset = set_data[1]
        print('mu_subset=', mu_subset, ' sig_subset=', sig_subset)
        # ---------------------------------------------------------------------
        print("Running the first level fit")
        moments_init = np.array([mu_subset, sig_subset])
        # Weighting matrix
        W_hat = np.eye(2)
        results = opt.minimize(
            self._criterion, moments_init, args=(self.subset, W_hat),
            method='L-BFGS-B',
            bounds=((1e-10, None), (1e-10, None)))
        mu_GMM1, sig_GMM1 = results.x
        print('mu_GMM1=', mu_GMM1, ' sig_GMM1=', sig_GMM1)
        # ---------------------------------------------------------------------
        print("Re-Running the fit with improved errors")
        start_t = time()
        # Improving the weighting matrix
        err1 = self._err_vec(self.subset, results.x, simple=False)
        # Need to reshape for the matrix
        err1 = np.array([[err1[0]], [err1[1]]])
        VCV2 = np.dot(err1, err1.T) / len(self.subset)
        # print(VCV2)
        # Use the pseudo-inverse calculated by SVD because
        # VCV2 is ill-conditioned
        W_hat2 = lin.pinv(VCV2)
        # print(W_hat2)
        # Re-fitting
        moments_init = np.array([mu_GMM1, sig_GMM1])
        results = opt.minimize(
            self._criterion, moments_init, args=(self.subset, W_hat2),
            method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
        mu_GMM2, sig_GMM2 = results.x
        print('mu_GMM2=', mu_GMM2, ' sig_GMM2=', sig_GMM2)
        end_t = time()
        print("GMM took %e seconds" % (end_t - start_t))
        # ---------------------------------------------------------------------
        print("Running scipy to compare")
        start_t = time()
        mu_scipy, sig_scipy = stats.norm.fit(self.subset)
        end_t = time()
        print("Norm fit took %e seconds" % (end_t - start_t))
        print('mu_scipy=', mu_scipy, ' sig_GMM2=', sig_scipy)
        return np.array([
            [mu_subset, sig_subset],
            [mu_GMM1, sig_GMM1],
            [mu_GMM2, sig_GMM2],
            [mu_scipy, sig_scipy]
        ])

    # Underlying function
    def _gaussian(self, x: np.array, mu: float, sigma: float) -> np.array:
        """ Returns the normal distribution function

        Parameters
        ----------
        x : np.array
            Array at which points to evaluate
        mu : float
            The mean value
        sigma : float
            The std of the distribution

        Returns
        -------
        res : np.array
            Numpy array with the evaluated points
        """
        res = (
            1. / (sigma * np.sqrt(2 * np.pi)) *
            np.exp(-(x - mu)**2 / (2 * sigma**2))
        )
        return res

    def _get_moments(self, data: np.array, order=2) -> np.array:
        """ Returns the moments up to order
        as a numpy array

        Parameters
        ----------
        data : np.array
            Array of the data (1D)
        order : int
            Optional: Order until which to calculate

        Returns
        -------
        moments : np.array
            Numpy array with length (order)
        """
        if type(order) != int:
            ValueError("Order must be of type int!")
        if order < 1:
            ValueError("Order must be greater or equal 1")
        moments = np.array([
            stats.moment(data, moment=i)
            if i > 1 else
            np.mean(data)
            for i in range(order+1)[1:]
        ])
        return moments

    def _model_moments(self, moments: np.array) -> np.array:
        """ Computes the moments of the model (here a gaussian)

        Parameters
        ----------
        moments : np.array
            The moments to use (gained from data)
        """
        def xfx(x):
            return x * self._pdf(x, moments[0], moments[1])
        (mean_model, _) = quad(
            xfx,
            moments[0] - 10 * moments[0] * moments[1],
            moments[0] + 10 * moments[0] * moments[1]
        )

        def x2fx(x):
            return (
                ((x - mean_model) ** 2) * self._pdf(x, moments[0], moments[1])
            )
        (var_model, _) = quad(
            x2fx,
            moments[0] - 10 * moments[0] * moments[1],
            moments[0] + 10 * moments[0] * moments[1]
        )

        return np.array([mean_model, var_model])

    def _err_vec(
            self, data: np.array, moments: np.array, simple=True) -> np.array:
        """ Calculates the error vector

        Parameters
        ----------
        data : np.array
            The data points to use
        moments : np.array
            The moments to estimate the error for
        simple : boolean
            If true: Simple difference, False: percentile deviations

        Returns
        -------
        error_vec : np.array
            Numpy array containing the error vector
        """
        data_moments = self._get_moments(data, order=2)
        m_moments = self._model_moments(moments)
        if simple:
            error_vec = m_moments - data_moments
        else:
            error_vec = (m_moments - data_moments) / data_moments

        return error_vec

    def _criterion(
            self, moments: np.array,
            data: np.array, W_hat: np.array) -> np.array:
        """ Function to minimize over

        Parameters
        ----------
        moments : np.array
            The moments to plug in the minimization function
        data : np.array
            The data to minimize against
        W_hat  : np.array
            Estimate of optimal weighting matrix

        Returns
        -------
        critical_val : np.array
            The measure values
        """
        err = self._err_vec(data, moments, simple=False)
        critical_val = np.dot(np.dot(err.T, W_hat), err)
        return critical_val

    def rebin(
            self,
            bin_content: np.array,
            old_grid: np.array,
            new_grid_edges: np.array,
            binning_scheme="Log",
            negatives=False):
        """ Rebins the binned counts to the new desired grid. This function
        uses a method of moments approach. Currently using 3 moments.

        Parameters
        ----------
        bin_contents: np.array
            The binned content which should be rebinned.
        old_grid: np.array
            The old grid's midpoints. The shape needs to be the same
            as bin_contents.
        new_grid_edges: np.array
            The new bins to use. These are the edeges of the grid.
        binning_scheme: str
            The binning scheme to use. Choices are "Log" (logarithmic)
            or "Lin" (linear). This decides how to calculate the midpoints
            of each bin.
        negatives: bool
            Switch to keep or remove negative values in the final binning.

        Returns
        -------
        new_content: np.array
            The new binned counts for the grid
        new_grid: np.array
            The bin midpoints used
        new_widths: np.array
            The bin widths used
        new_edges: np.array
            The edges used. These are the same as the ones passed

        Warnings
        --------
        ValueError:
            Unknown binning scheme

        Notes
        -----
        TODO:
            1) Expand to multiple dimensions
            2) Deal with edge cases for the new grid
            3) Fix the normalization. Unified approach for pure positive
               would be neat.
        """
        # Setting up the new grid
        if binning_scheme == "Log":
            new_grid = np.sqrt(new_grid_edges[1:] * new_grid_edges[:-1])
        elif binning_scheme == "Lin":
            new_grid = (new_grid_edges[1:] + new_grid_edges[:-1]) / 2.
        else:
            ValueError("Unknown binning scheme! 'Log' and 'Lin' supported")
        new_widths = new_grid[1:] - new_grid[:-1]
        new_edges = new_grid_edges
        # Checking if shapes align
        if bin_content.shape != old_grid.shape:
            ValueError("Bin content and old grid do not have the same shape!")
        # Unfilled new list
        new_content = np.zeros(new_grid.shape)
        # Looping over the old bin contents and distributing
        for id_val, bin_val in enumerate(bin_content):
            # Ignore bins without values
            if bin_val == 0.:
                continue
            tmp_grid_val = old_grid[id_val]
            new_point = (np.abs(new_grid - tmp_grid_val)).argmin()

            # Setting up the equation for 3 moments (mat*x = b)
            # x is the values we want
            # NOTE: +2 since for upper bounds the value is not used
            mat = np.vstack(
                (
                    new_widths[new_point - 1:new_point + 2],
                    new_widths[new_point - 1:new_point + 2] *
                    new_grid[new_point - 1:new_point + 2],
                    new_widths[new_point - 1:new_point + 2] *
                    new_grid[new_point - 1:new_point + 2]**2
                )
            )
            b = bin_val * np.array([
                1.,
                tmp_grid_val,
                tmp_grid_val**2
            ])
            # Solving and adding to the new content
            tmp_new_content = solve(
                mat, b
            )
            new_content[new_point - 1:new_point + 2] += tmp_new_content
        if not negatives:
            new_content[new_content < 0.] = 0.
        # TODO: Remove this dependency
        new_content = (
            new_content / (np.sum(new_content) / np.sum(bin_content))
        )
        return new_content, new_grid, new_widths, new_edges
