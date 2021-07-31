import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def analytical_downstream(xi, B=0.321):
    return 1 - B*np.cos(np.sqrt(3)/2*xi)*np.exp(xi/2)

data = np.loadtxt('analytical/literature/lhuisser2013_fig4b.txt', delimiter='\t', skiprows=1)[3:,:]
plt.plot(data[:,0], data[:,1], color='black')
# dx = data[1,0]-data[0,0]
# dydx = np.gradient(data[:,1], dx)
# d2ydx2 = np.gradient(dydx, dx)
# xi0 = data[0,0]
# H0 = data[0,1]
# dy_0 = dydx[0]
# d2y_0 = d2ydx2[0]

xs = np.linspace(-33.25,2,10000)
dx = xs[1]-xs[0]
ys = analytical_downstream(xi=xs)
dydx = np.gradient(ys, dx)
d2ydx2 = np.gradient(dydx, dx)
plt.plot(xs, ys)
plt.plot(xs, d2ydx2)
# plt.show()



# define the ODE as a first order system
def func(y, x):
    return [
        y[1],
        y[2],
        (1 - y[0])/y[0]**3
        ]

# d2y_0 = 2e-7
# initial values
y0=[ ys[0], dydx[0], d2ydx2[0]]
# points at which the solution value is requested
x = np.linspace(xs[0],10,100000)
# numerical integration
additiona_precision = 1e-5
y=odeint(func, y0, x, rtol=1.49012e-8*additiona_precision, atol=1.49012e-8*additiona_precision)
# y[-1,:] contains the value at x=10
# print "[ y(10), y'(10), y''(10) ] = ", y[-1,:]
print(y[-1,2])
plt.plot(x, y[:,0])
plt.plot(x, y[:,2], linestyle='--')

dx = x[1]-x[0]
dydx = np.gradient(y[:,0], dx)
d2ydx2 = np.gradient(dydx, dx)

plt.plot(x, d2ydx2, linestyle='--')

plt.show()

# # plot the solution with subplots for each component
# fig=plt.figure()
# ax=fig.add_subplot(311)
# ax.plot(x, y[:,0])
# ax=fig.add_subplot(312)
# ax.plot(x, y[:,1])
# ax=fig.add_subplot(313)
# ax.plot(x, y[:,2])
# plt.show()