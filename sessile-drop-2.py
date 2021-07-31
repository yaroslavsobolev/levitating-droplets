import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.interpolate import UnivariateSpline
from scipy.integrate import trapz
from scipy.optimize import curve_fit
from scipy.optimize import fmin
from scipy.signal import savgol_filter
# from shapely.geometry import Polygon
import pickle

def get_curvature(x, y, wlen = 3):
    # t = np.arange(x.shape[0])
    xˈ = savgol_filter(x, polyorder=1, window_length=wlen, deriv=1)
    xˈˈ = savgol_filter(x, polyorder=2, window_length=wlen, deriv=2)
    yˈ = savgol_filter(y, polyorder=1, window_length=wlen, deriv=1)
    yˈˈ = savgol_filter(y, polyorder=2, window_length=wlen, deriv=2)
    curvature = (xˈ * yˈˈ - yˈ * xˈˈ) / np.power(xˈ ** 2 + yˈ ** 2, 1.5)
    return curvature

def max_curvature(x, y):
    curvature = get_curvature(x,y)
    curvature = np.abs(curvature)[1:-1]
    curvmax1 = np.max(curvature)
    return curvmax1

def distance(P1, P2):
    """
    This function computes the distance between 2 points defined by
     P1 = (x1,y1) and P2 = (x2,y2)
    """
    return ((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2) ** 0.5

def optimized_path(coords, start=None):
    """
    This function finds the nearest point to a point
    coords should be a list in this format coords = [ [x1, y1], [x2, y2] , ...]

    """
    if start is None:
        start = coords[0]
    pass_by = coords
    path = [start]
    pass_by.remove(start)
    while pass_by:
        nearest = min(pass_by, key=lambda x: distance(path[-1], x))
        path.append(nearest)
        pass_by.remove(nearest)
    return path

rho = 1000 # kg/m^3
sigma0 = 72e-3 # N/m
g_const = 9.8067 # m/s^2
shapes = []
do_plot = False

def caplen(sigma):
    return np.sqrt(sigma/(g_const*rho))



# load sigmas
sigma_factors = np.loadtxt('comsol_results/sessile-drop/sigma_piecewise.txt')[:,2]

shapes = pickle.load(open("analytical/sessile-drop/shapes.p", "rb") )


#
# for shape in shapes:
#     _, _, a, _, volume, curve = shape
#     a = a*1e3
#     curve = curve/a
#     curve = np.append(curve, [np.array([0, 0])], axis=0)
#     plt.plot(curve[:,0], curve[:,1], color='black', alpha=0.3, solid_capstyle="butt")
#     plt.plot(-curve[:, 0], curve[:, 1], color='black', alpha=0.3, solid_capstyle="butt")
#     # plt.fill(curve[:,0], curve[:,1], color='black', alpha=0.1, linewidth=0)
#
# # plt.axhline(y=0, linestyle='--', color='C0')
# plt.axhspan(ymin=-5, ymax=0, facecolor='C2', alpha=0.4)
# plt.axis('equal')
# plt.xlabel('Horizontal coordinate, dimensionless $x/a$')
# plt.ylabel('Vertical coordinate,\ndimensionless $z/a$')
# plt.show()
# print(times)
#
# plt.plot([x[1] for x in shapes], [x[4] for x in shapes], 'o-')
# plt.show()
#
# # for shape in shapes:
# #     _, _, a, _, volume, curve = shape
# #     plt.scatter(a, volume/(4/3*np.pi*(np.max(curve[:,1])/2)**3))
# # plt.show()
#
# for shape in shapes:
#     _, _, a, maxcurv, volume, curve = shape
#     r1 = np.max(curve[:,1])/2
#     r2 = (volume*3/4/np.pi)**(1/3)
#     plt.scatter(a, maxcurv/(a*1000/r2))
# plt.show()
#
# ka = np.array([x[3] for x in shapes])
# Vs = np.array([x[4]/(x[2]*1000)**3 for x in shapes])
# plt.plot(Vs, ka, 'o-')
# plt.show()
#
# xdata = 1/Vs
# ydata = ka**2
# plt.loglog(xdata, ydata, 'o-')
#
# def func(x, a, b, c):
#     return a*x**b + c
#
# pref1 = 2.5985
# popt, pcov = curve_fit(func, xdata[-20:], ydata[-20:], p0=(8, 2/3, 4), bounds=[(0, 2/3-0.001, 4-0.001),
#                                                                        (np.inf, 2/3+0.001, 4+0.001)],
#                        sigma=ydata[-20:], ftol=1e-30)
# print(popt)
# plt.plot(xdata, func(xdata, *popt), 'r-',
#          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
# plt.grid()
# plt.show()