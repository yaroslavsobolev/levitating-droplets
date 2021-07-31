import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import optimize

def analytical_downstream_derivs(xi, B=-0.321):
    # these are from [Lhuisser 2013], formula (3.10) with considerations on the next page (page R4-7)
    R = np.sqrt(3)/2
    # the function itself
    y = 1 + B*np.cos(R*xi)*np.exp(xi/2)
    # first derivative
    yy = B * np.exp(xi / 2) * ( 1/2*np.cos(R * xi) - R * np.sin(R * xi) )
    # second derivative
    yyy = B * np.exp(xi / 2) * ( (1 / 4 - R**2) * np.cos(R * xi) - R * np.sin( R * xi))
    return [y, yy, yyy]

def analytical_upstream_derivs(xi, A=1):
    # these are from [Lhuisser 2013], formula (3.10) with B=0
    # the function itself
    y = 1 + A*np.exp(-xi)
    # first derivative
    yy = -1*A*np.exp(-xi)
    # second derivative
    yyy = A*np.exp(-xi)
    return [y, yy, yyy]

def numerical_downstream(x0 = -33.25, B=-0.321, xmax=20, W=0, do_plot=False):
    # print(B)
    # define the ODE as a first order system
    def func(y, x):
        return [
            y[1],
            y[2],
            (1 - y[0] - W*y[1])/y[0]**3
            ]

    # initial values for function y, its first derivative, and its second derivative
    y0 = analytical_downstream_derivs(xi=x0, B=B)

    # points at which the solution value is requested
    x = np.linspace(x0,xmax,200000)

    # numerical integration
    additiona_precision = 1e-5
    y=odeint(func, y0, x, rtol=1.49012e-8*additiona_precision, atol=1.49012e-8*additiona_precision)

    if do_plot:
        data = np.loadtxt('analytical/literature/lhuisser2013_fig4b.txt', delimiter='\t', skiprows=1)[3:, :]
        plt.plot(data[:, 0], data[:, 1], color='black')
        print(y[-1,2])
        plt.plot(x, y[:,0])
        plt.plot(x, y[:,2], linestyle='--')

    return x, y, y[-1,2]

def find_B_to_match_curvature(target_curvature = 0.643, x0 = -33.25, B0=-0.321, xmax=20, W=0):
    def fun(B):
        _, _, curvature_at_inf = numerical_downstream(x0 = x0, B=B, xmax=xmax, W=W, do_plot=False)
        return curvature_at_inf - target_curvature
    root = optimize.brentq(fun, a=B0*0.98, b=B0*1.3)
    return root

def numerical_upstream(x0 = 20, A=1, xmin=-20, W=0, do_plot=False):
    # print(B)
    # define the ODE as a first order system
    def func(y, x):
        return [
            y[1],
            y[2],
            (1 - y[0] - W*y[1])/y[0]**3
            ]

    # initial values for function y, its first derivative, and its second derivative
    y0 = analytical_upstream_derivs(xi=x0, A=A)

    # points at which the solution value is requested
    x = np.linspace(x0,xmin,200000)

    # numerical integration
    additiona_precision = 1e-5
    y=odeint(func, y0, x, rtol=1.49012e-8*additiona_precision, atol=1.49012e-8*additiona_precision)

    if do_plot:
        # data = np.loadtxt('analytical/literature/lhuisser2013_fig4b.txt', delimiter='\t', skiprows=1)[3:, :]
        # plt.plot(data[:, 0], data[:, 1], color='black')
        # print(y[-1,2])
        plt.plot(x, y[:,0])
        plt.plot(x, y[:,2], linestyle='--',alpha=0.5)

    return x, y, y[-1,2]

def upstream_curvature(W):
    x, y, curv = numerical_upstream(x0=20, A=1, xmin=-150, W=W, do_plot=False)
    return curv

def make_upstream_curvature_plot():
    # find upstream curvature for different W
    data = np.loadtxt('analytical/literature/lhuisser2013_fig4a.txt', delimiter='\t', skiprows=1)[:, :]
    plt.plot(data[:, 0], data[:, 1], color='black', label='Literature solution')
    x,y,curv = numerical_upstream(x0 = 20, A=1, xmin=-150, W=3, do_plot=True)
    print(curv)
    plt.show()
    curvs = []
    Ws = np.linspace(0, 5e-1*12, 30*12)
    for i,W in enumerate(Ws):
        x, y, curv = numerical_upstream(x0=20, A=1, xmin=-150, W=W, do_plot=False)
        curvs.append(curv)
    np.save('analytical/ws.npy', Ws)
    np.save('analytical/curvs.npy', np.array(curvs))
    plt.plot(Ws, curvs)
    plt.show()

# _, _, curvature_at_inf = numerical_downstream(do_plot=True)
# print(curvature_at_inf)


def shape_for_W(W, B_prev, do_plot=True):
    x0 = -50
    xmax = 40
    B = find_B_to_match_curvature(x0=x0, W=W, B0=B_prev, target_curvature=upstream_curvature(W),
                                  xmax=xmax)
    x,y,curv = numerical_downstream(x0=x0, B=B, W=W, do_plot=False, xmax=xmax)
    if do_plot:
        plt.plot(x[::20], y[::20, 0], alpha=0.5, label='W={0:.3f}'.format(W))
    return B, np.min(y[:,0])


# data = np.loadtxt('analytical/literature/lhuisser2013_fig4b.txt', delimiter='\t', skiprows=1)[3:, :]
# plt.plot(data[:, 0], data[:, 1], color='black')
# W=0
# x0 = -50
# B = find_B_to_match_curvature(x0=x0, W=W, B0=-0.33, target_curvature=upstream_curvature(W))
# x, y, curv = numerical_downstream(x0=x0, B=B, xmax=20, W=W, do_plot=True)
# print(B)
# plt.show()

data = np.loadtxt('analytical/literature/lhuisser2013_fig4b.txt', delimiter='\t', skiprows=1)[3:, :]
plt.plot(data[:, 0], data[:, 1], color='black', label='Literature solution')
# print(y[-1, 2])
B_prev = -0.33
Hmins = []
Ws = np.linspace(0, 5e-1, 100)
for i,W in enumerate(Ws):
    B_prev, Hmin = shape_for_W(W, B_prev, do_plot=(i % 10 == 0))
    Hmins.append(Hmin)
    print('W={0}, B={1}'.format(W,B_prev))
plt.legend()
plt.xlim(-20, 3)
plt.ylim(0, 3)
# plt.plot(x, y[:, 2], linestyle='--')

plt.show()

plt.plot(Ws, Hmins)
plt.plot()
plt.show()