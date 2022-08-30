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
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def colorline(x, y, dydx, fig, ax):
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    return line

def get_curvature(x, y, wlen = 3):
    # t = np.arange(x.shape[0])
    xˈ = savgol_filter(x, polyorder=2, window_length=wlen, deriv=1)
    xˈˈ = savgol_filter(x, polyorder=2, window_length=wlen, deriv=2)
    yˈ = savgol_filter(y, polyorder=2, window_length=wlen, deriv=1)
    yˈˈ = savgol_filter(y, polyorder=2, window_length=wlen, deriv=2)
    curvature = (xˈ * yˈˈ - yˈ * xˈˈ) / np.power(xˈ ** 2 + yˈ ** 2, 1.5)
    return curvature

def max_curvature(x, y):
    curvature = get_curvature(x,y)
    curvature = np.abs(curvature)[2:-2]
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

def convert_from_comsol(N=10):
    for nfile in range(N):
        # get times from header
        target_file = "comsol_results/sessile-drop/shape_data{0:02d}.txt".format(nfile)
        fp = open(target_file)
        for i, line in enumerate(fp):
            if i == 8:
                header = line
            elif i>8:
                break
        fp.close()
        print(header)
        regex = re.compile(r'''dom @ t=(?P<time>\S+) ''')
        res = regex.findall(header)
        times = [float(r) for r in res]
        data = np.loadtxt(target_file, skiprows=9)

        # get nth time:
        for N, time in enumerate(times):
            sigma = sigma0*sigma_factors[int(round(time // 10))]
            col_nums = [3*N + 2, 3*N + 3, 3*N + 4]
            shape_here = data[:,col_nums]
            doms = shape_here[:,0]
            mask = (doms == 4) | (doms == 6) | (doms == 8) | (doms == 10)
            shape_here = shape_here[mask,1:]
            startindex = np.argmin(shape_here[:,0])
            print('Startindex:{0}'.format(startindex))
            opshape = np.array(optimized_path(shape_here.tolist(), start=[shape_here[startindex, 0],
                                                                          shape_here[startindex, 1]]))
            # curvatures = curvature_splines(opshape[:, 0], opshape[:, 1])
            max_curv_unitless = max_curvature(opshape[:, 0], opshape[:, 1])*caplen(sigma)*1000
            # volume
            volume = np.abs(trapz(np.pi*opshape[:,0]**2, opshape[:,1]))
            print([caplen(sigma), max_curv_unitless])
            if do_plot:
                # plt.plot(curvatures, 'o-')
                plt.show()
                plt.scatter(shape_here[0, 0], shape_here[0, 1])
                # plt.plot(shape_here[:,0], shape_here[:,1])
                plt.scatter(opshape[:, 0], opshape[:, 1])
                plt.axis('equal')
                plt.colorbar()
                plt.show()
            shapes.append([time, sigma, caplen(sigma), max_curv_unitless, volume, opshape])

    pickle.dump(shapes, open("analytical/sessile-drop/shapes.p", "wb") )
    print('Pickle dumped')

if __name__ == '__main__':
    # convert_from_comsol() # UNCOMMENT FOR PRODUCING "analytical/sessile-drop/shapes.p" FROM RAW COMSOL DATA FILES

    shapes = pickle.load(open("analytical/sessile-drop/shapes.p", "rb") )

    for shape in shapes:
        _, _, a, _, volume, curve = shape
        a = a*1e3
        curve = curve/a
        curve = np.append(curve, [np.array([0, 0])], axis=0)
        plt.plot(curve[:,0], curve[:,1], color='black', alpha=0.3, solid_capstyle="butt")
        plt.plot(-curve[:, 0], curve[:, 1], color='black', alpha=0.3, solid_capstyle="butt")
        # plt.fill(curve[:,0], curve[:,1], color='black', alpha=0.1, linewidth=0)

    # plt.axhline(y=0, linestyle='--', color='C0')
    plt.axhspan(ymin=-5, ymax=0, facecolor='C2', alpha=0.4)
    plt.axis('equal')
    plt.xlabel('Horizontal coordinate, dimensionless $x/a$')
    plt.ylabel('Vertical coordinate,\ndimensionless $z/a$')
    plt.show()

    plt.plot([x[1] for x in shapes], [x[4] for x in shapes], 'o-')
    plt.show()

    # for shape in shapes:
    #     _, _, a, _, volume, curve = shape
    #     plt.scatter(a, volume/(4/3*np.pi*(np.max(curve[:,1])/2)**3))
    # plt.show()

    for shape in shapes:
        _, _, a, maxcurv, volume, curve = shape
        r1 = np.max(curve[:,1])/2
        r2 = (volume*3/4/np.pi)**(1/3)
        plt.scatter(a, maxcurv/(a*1000/r2))
    plt.show()

    # fit the dependence of kappa_b on the volume
    f_kb_fit = plt.figure(figsize=(4.3, 3.5))
    ka = np.array([x[3] for x in shapes])
    Vs = np.array([x[4]/(x[2]*1000)**3 for x in shapes])
    # plt.plot(Vs, ka, 'o-')
    # plt.show()

    xdata = 1/(Vs*3/4/np.pi)
    ydata = ka**2
    plt.loglog(xdata, ydata, 'o-', label='From drop shape found by FEM')

    def func(x, a, b, c):
        return a*x**b + c

    pref1 = 2.5985
    # bounds = [(0, 2 / 3 - 0.001, 4 - 0.001), (np.inf, 2 / 3 + 0.001, 4 + 0.001)]
    # bounds = [(0, 2 / 3 - 0.001, 0), (np.inf, 2 / 3 + 0.001, np.inf)]
    # bounds = [(0, 2 / 3 - 0.001, 0), (np.inf, 2 / 3 + 0.001, np.inf)]
    bounds = [(0, 0.5, 0), (np.inf, 3, np.inf)]
    p0 = (4, 2 / 3, 4)
    bounds = [p0, [x+0.001 for x in p0]]
    from_n = 0
    to_n = -1
    popt, pcov = curve_fit(func, xdata[from_n:to_n], ydata[from_n:to_n], p0=p0, bounds=bounds,
                           sigma=ydata[from_n:to_n]**0.8, ftol=1e-30)
    print(popt)
    plt.plot(xdata, func(xdata, *popt), 'r-',
             label='Analytical, $(k_b a)^2 = 4\hat{V}^{-2/3} + 4$')
             # label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.grid()
    plt.legend()
    plt.xlabel('1/$\hat{V}$')
    plt.ylabel('$(k_b a)^2$')
    plt.tight_layout()
    f_kb_fit.savefig('figures/kb_vs_V_fit.png', dpi=300)
    plt.show()

    # plot one of the shapes
    time, sigma, a, max_curv_unitless, volume, shape = shapes[20]
    curv = get_curvature(shape[:, 0], shape[:, 1])
    plt.plot(np.abs(curv), 'o-')
    rad = np.max(shape[:, 1]) / 2
    plt.axhline(y=1 / rad)
    plt.show()
    fig, ax = plt.subplots()
    colorline(shape[:, 0], shape[:, 1], curv, fig, ax)
    rightmost_index = np.argmax(shape[:, 0])
    plt.scatter(shape[rightmost_index, 0], shape[rightmost_index, 1])
    plt.axis('equal')
    plt.show()

    rightmost_curvs = []
    max_curvs = []
    for s in shapes:
        time, sigma, a, max_curv_unitless, volume, shape = s
        curv = get_curvature(shape[:, 0], shape[:, 1])
        rightmost_index = np.argmax(shape[:, 0])
        rightmost_curvs.append((a * 1000) * np.abs(curv[rightmost_index]))
        max_curvs.append((a * 1000) * np.max(np.abs(curv)))

    rightmost_curvs = np.array(rightmost_curvs)
    max_curvs = np.array(max_curvs)
    plt.plot(rightmost_curvs ** 2, max_curvs ** 2)
    plt.plot(rightmost_curvs ** 2, rightmost_curvs ** 2 + 2, '--')
    plt.show()