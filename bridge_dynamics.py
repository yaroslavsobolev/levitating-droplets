import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate
from scipy.integrate import quad
import pickle

target_folder = 'comsol_results/capillary_bridge_dynamics/sketch_1/'
comsol_filename = target_folder + 'velocity_field.txt'
comsol_npy_filename = target_folder + 'velocity_field.npy'
interpolators_filename = target_folder + 'interps.pkl'

# comsol_data = np.loadtxt(comsol_filename, skiprows=9)
# np.save(comsol_npy_filename, comsol_data)

comsol_data = np.load(comsol_npy_filename)
r = comsol_data[:,0]
z = comsol_data[:,1]

line = open(comsol_filename, "r").readlines()[8]
regex1 = re.compile('''@ t=(.+?) ''', re.DOTALL)
res = regex1.findall(line)
ts = np.array(sorted([float(x) for x in list(set(res))]))

def get_field_for_t(t):
    index = np.where(ts == t)[0][0]
    #u,w
    u = comsol_data[:, 2 + index * 3]
    w = comsol_data[:, 2 + index * 3 + 1]
    return u,w

def get_field_interp_for_timenode(t):
    u,w = get_field_for_t(t)
    # position = np.argmax(ts > t)

    x = r
    y = z
    Ex_raw = u
    Ey_raw = w

    xi = np.array(sorted(list(set(list(x)))))
    yi = np.array(sorted(list(set(list(y)))))
    X, Y = np.meshgrid(xi, yi)

    Ex = griddata((x, y), Ex_raw, (X, Y), method='linear')
    Ey = griddata((x, y), Ey_raw, (X, Y), method='linear')

    # plt.pcolor(X,Y,Ey)
    # plt.colorbar()
    # plt.show()

    # plt.plot(xi, Ey[0,:], color='C0')
    # plt.show()

    Ex_interp = RegularGridInterpolator(points=[xi, yi], values=Ex.T, method='linear', fill_value=None)
    Ey_interp = RegularGridInterpolator(points=[xi, yi], values=Ey.T, method='linear', fill_value=None)

    return Ex_interp, Ey_interp

def precompute_interps():
    print('Precomputing interpolators.')
    interp_list = [get_field_interp_for_timenode(t) for t in ts]
    with open(interpolators_filename, 'wb') as f:
        pickle.dump(interp_list, f)

def load_interps():
    with open(interpolators_filename, 'rb') as f:
        interp_list = pickle.load(f)
    return interp_list

# precompute_interps()
interp_list = load_interps()

def get_interface_line_for_timenode(t):
    index = np.where(ts == t)[0][0]
    dom = comsol_data[:, 2 + index * 3 + 2]
    ri = np.array(sorted(list(set(list(r)))))
    zi = np.array(sorted(list(set(list(z)))))
    R, Z = np.meshgrid(ri, zi)
    dom_grid = griddata((r, z), dom, (R, Z), method='linear')
    interface_rs = []
    for z_index, z_here in enumerate(zi):
        if np.any(dom_grid[z_index, :] > 1):
            r_index = np.argmax(dom_grid[z_index, :] > 1)
        else:
            r_index = -1
        interface_rs.append(ri[r_index])
    interface_rs = np.array(interface_rs)
    plt.plot(interface_rs, zi, '-', label='{0}'.format(t), color='black')

# get_interface_line_for_timenode(ts[70])
# get_interface_line_for_timenode(ts[100])
# plt.legend()
# plt.show()

def sample_field_interp(points, Ex_interp, Ey_interp):
    return np.stack((Ex_interp(points), Ey_interp(points))).T

def field_at_arbitrary_point(r0,z0,t0, precomputed=True):
    right_time_index = np.argmax(ts > t0)
    left_time_index = right_time_index - 1
    t1 = ts[left_time_index]
    t2 = ts[right_time_index]

    point = np.array([r0, z0])

    if not precomputed:
        Ex_interp, Ey_interp = get_field_interp_for_timenode(t1)
        u1, w1 = sample_field_interp(point, Ex_interp, Ey_interp)[0]
        Ex_interp, Ey_interp = get_field_interp_for_timenode(t2)
        u2, w2 = sample_field_interp(point, Ex_interp, Ey_interp)[0]
    else:
        u1, w1 = sample_field_interp(point, interp_list[left_time_index][0], interp_list[left_time_index][1])[0]
        u2, w2 = sample_field_interp(point, interp_list[right_time_index][0], interp_list[right_time_index][1])[0]


    u_interp = interpolate.interp1d([t1, t2], [u1, u2])([t0])[0]
    w_interp = interpolate.interp1d([t1, t2], [w1, w2])([t0])[0]
    return u_interp, w_interp

# u_here, w_here = field_at_arbitrary_point(10, 50, ts[70])
# get_interface_line_for_timenode(ts[70])
# plt.show()


# get_field_interp_for_timenode(ts[0])
# ws = []
# for r_here in np.array(sorted(list(set(list(r))))):
#     u_here, w_here = field_at_arbitrary_point(r_here, 50.4, ts[0])
#     ws.append(np.copy(w_here))
# plt.plot(np.array(sorted(list(set(list(r))))), ws, color='C1')
# plt.show()

# pos = np.array([6.74, 50.098])
positions = []
rs = np.array(sorted(list(set(list(r)))))
for r_here in rs[:100]:
    positions.append([r_here, 50.02])
positions = np.array(positions)
pos_list = [positions]
# tss = np.logspace(-7, np.log10(ts[100]), num=100)
tss = ts
t_list = [tss[0]]
volume_list = [0]
for i in range(tss.shape[0]-1):
    for j in range(positions.shape[0]):
        u_here, w_here = field_at_arbitrary_point(positions[j, 0], positions[j, 1], tss[i])
        positions[j, :] += np.array([u_here, w_here])*1e6*(tss[i+1]-tss[i])
    t_list.append(tss[i+1])
    pos_list.append(np.copy(positions))
    print(i)
    print(tss[i])
    f_interp = lambda rr: np.interp(rr, positions[:, 0], positions[:, 1])
    def f_volume(rr):
        return f_interp(rr)*2*np.pi*rr
    tip = np.argmax(positions[:, 1])
    tipr = positions[tip, 0]
    tipz = positions[tip, 1]
    volume = np.pi*tipr**2*np.max(positions[:, 1]) - quad(f_volume, 0, tipr)[0]
    print('Volume: {0}'.format(volume))
    volume_list.append(volume)
    if i > 130:
        break
    # if i % 5 == 0:
    #     fig = plt.figure(1)
    #     u, w = get_field_for_t(ts[i+1])
    #     xi = np.array(sorted(list(set(list(r)))))
    #     yi = np.array(sorted(list(set(list(z)))))
    #     R, Z = np.meshgrid(xi, yi)
    #     Ey = griddata((r, z), w, (R, Z), method='linear')
    #     plt.pcolor(R,Z,Ey, cmap='seismic', vmin=-1*np.max(np.abs(Ey)), vmax=np.max(np.abs(Ey)))
    #     plt.colorbar()
    #     get_interface_line_for_timenode(ts[i+1])
    #     plt.plot(positions[:, 0], positions[:, 1], '-', color='green')
    #     plt.scatter(tipr, tipz, color='yellow')
    #     plt.xlim(0, 20)
    #     plt.ylim(50, 52)
    #     fig.savefig('comsol_results/capillary_bridge_dynamics/figures/frame{0:08d}.png'.format(i))
    #     # plt.show()
    #     plt.close('all')

for i in range(len(volume_list)):
    plt.loglog(tss[i], volume_list[i], 'o-')
plt.show()
# # pos_list = np.array(pos_list)
# for positions in pos_list:
#     plt.plot(positions[:,0], positions[:,1], '-')
# get_interface_line_for_timenode(ts[0])
# get_interface_line_for_timenode(ts[100])
# plt.show()

print(1)