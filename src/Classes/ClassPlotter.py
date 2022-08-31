import numbers

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class Plotter:
    """ Abstract class for plotting QQ plots and histograms"""

    def __init__(self, x1data, y1data, xlabel="", ylabel="", plotlabel="", plottitle="", fig=None,
                 ax=None):
        self.__x1 = x1data
        self.__y1 = y1data
        self.__xlabel = xlabel
        self.__ylabel = ylabel
        self.__plotlabel = plotlabel
        self.__plottitle = plottitle
        if ax and fig:
            self.fig, self.ax = fig, ax
        else:
            self.fig, self.ax = plt.subplots()

    def get_x1(self):
        return self.__x1

    def get_y1(self):
        return self.__y1

    def get_xlabel(self):
        return self.__xlabel

    def get_ylabel(self):
        return self.__ylabel

    def get_plotlabel(self):
        return self.__plotlabel

    def get_plottitle(self):
        return self.__plottitle

    def set_x1(self, new_x1):
        self.__x1 = new_x1

    def set_y1(self, new_y1):
        self.__y1 = new_y1

    def plot(self):
        return


class HistogramPlotter(Plotter):
    def __init__(self, x1samples, y1samples=None, xlabel="", ylabel="", plotlabel="", plottitle="", bins=200,
                 pdf_values=None, fig=None, ax=None):
        super().__init__(x1samples, y1samples, xlabel=xlabel, ylabel=ylabel, plotlabel=plotlabel, plottitle=plottitle,
                         fig=fig, ax=ax)
        self.__pdf_vals = pdf_values
        self.__bins = bins

    def get_pdf_vals(self):
        return self.__pdf_vals

    def plot(self):
        """ Function to compare generated process with density at t = T """
        self.fig.set_size_inches(14, 9.5)
        x1 = self.get_x1()
        self.ax.set_xlabel(self.get_xlabel())
        self.ax.set_ylabel(self.get_ylabel())
        self.ax.set_title(self.get_plottitle())
        binvals, _, _ = plt.hist(x1, self.__bins, density=True, label="Histogram of Process at $t = T_{horizon}$")
        self.ax.plot(np.linspace(min(x1), max(x1), len(x1)), self.__pdf_vals, label=self.get_plotlabel())
        self.ax.legend()

    def plot_normal(self):
        """ Function to compare generated process with density at t = T """
        self.fig.set_size_inches(14, 9.5)
        x1 = self.get_x1()
        self.ax.set_xlabel(self.get_xlabel())
        self.ax.set_ylabel(self.get_ylabel())
        self.ax.set_title(self.get_plottitle())
        binvals, _, _ = plt.hist(x1, self.__bins, density=True, label="Histogram of Process at $t = T_{horizon}$")
        xvals = np.linspace(norm.ppf(0.00001), norm.ppf(0.99999), x1.shape[0])
        self.ax.plot(xvals, self.__pdf_vals, label=self.get_plotlabel())
        self.ax.legend()



class QQPlotter(Plotter):
    def __init__(self, x1samples, y1samples, ylabel="", plotlabel="", plottitle="", log=True, x2samples=None,fig=None, ax=None):
        super().__init__(x1samples, y1samples, xlabel="Theoretical RVs", ylabel=ylabel, plotlabel=plotlabel,
                         plottitle=plottitle, fig=fig, ax=ax)
        self.__log = log
        self.__x2 = x2samples

    def plot(self, quantiles=None, interpolation='nearest', rug=False,
             rug_length=0.05, rug_kwargs=None, **kwargs):
        self.fig.set_size_inches(14, 9.5)
        x1 = self.get_x1()
        y1 = self.get_y1()
        xlabel = self.get_xlabel()
        ylabel = self.get_ylabel()

        if self.ax is None:
            self.ax = plt.gca()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(self.get_plottitle())
        if quantiles is None:
            quantiles = min(len(x1), len(y1))

        # Compute quantiles of the two samples
        if isinstance(quantiles, numbers.Integral):
            quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
        else:
            quantiles = np.atleast_1d(np.sort(quantiles))
        x_quantiles1 = np.quantile(x1, quantiles, interpolation=interpolation)
        y_quantiles1 = np.quantile(y1, quantiles, interpolation=interpolation)
        if self.__x2 is not None:
            if quantiles is None:
                quantiles = min(len(self.__x2), len(y1))

            # Compute quantiles of the two samples
            if isinstance(quantiles, numbers.Integral):
                quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
            else:
                quantiles = np.atleast_1d(np.sort(quantiles))
            x_quantiles2 = np.quantile(self.__x2, quantiles, interpolation=interpolation)
            y_quantiles2 = np.quantile(y1, quantiles, interpolation=interpolation)

        # Draw the rug plots if requested
        if rug:
            # Default rug plot settings
            rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
            rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

            # Override default setting by any user-specified settings
            if rug_kwargs is not None:
                rug_x_params.update(rug_kwargs)
                rug_y_params.update(rug_kwargs)

            # Draw the rug plots
            for point in x1:
                self.ax.axvline(point, **rug_x_params)
            for point in y1:
                self.ax.axhline(point, **rug_y_params)
            if self.__x2 is not None:
                for point in self.__x2:
                    self.ax.axvline(point, **rug_x_params)
                for point in y1:
                    self.ax.axhline(point, **rug_y_params)

        # Draw the q-q plot and compare with y = x
        self.ax.scatter(x_quantiles1, y_quantiles1, c="black", label="Q-Q plot", **kwargs)
        if self.__x2 is not None:
            self.ax.scatter(x_quantiles2, y_quantiles2, c='red', label="Q-Q plot", **kwargs)
        lims = [
            np.min([self.ax.get_xlim(), self.ax.get_ylim()]),  # min of both axes
            np.max([self.ax.get_xlim(), self.ax.get_ylim()]),  # max of both axes
        ]
        # now plot both limits against each other
        self.ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label="45 degree line")
        if self.__log:
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
        self.ax.legend()


class TimeSeriesPlotter(Plotter):
    def __init__(self, time_ax, yvals, xlabel="", ylabel="", plotlabel="", plottitle="", colour="black", fig=None, ax=None):
        super().__init__(time_ax, yvals, xlabel, ylabel, plotlabel, plottitle, fig=fig, ax=ax)
        self.__colour = colour

    def get_colour(self):
        return self.__colour

    def set_colour(self, new_colour):
        self.__colour = new_colour

    def plot(self):
        self.fig.set_size_inches(14, 9.5)
        self.ax.plot(self.get_x1(), self.get_y1(), linestyle='dashed', label=self.get_plotlabel())
        self.ax.set_xlabel(self.get_xlabel())
        self.ax.set_ylabel(self.get_ylabel())
        self.ax.set_title(self.get_plottitle())
        if self.get_plotlabel():
            self.ax.legend()
