import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


# load the boundary between the stable and unstable regions
from tqdm import tqdm

stability_boundary = np.loadtxt('misc_data/experimental_stability/region-boundary.txt', delimiter='\t', skiprows=1)
stability_boundary_interpolator = interp1d(stability_boundary[:, 1], stability_boundary[:, 0], kind='linear',
                                           fill_value='extrapolate')

threshold_viscosity = 0.0085 # Pa*s

def is_stable(x, y):
    return x > stability_boundary_interpolator(y)

def draw_map_for_all_frames(temperatures_txt_filepath='E:/levitating_droplet_with_heating/ir-sensor-videos/ir_sensor_temperatures.txt',
                            viscosity_arrhenius_parameters_file='misc_data/rheometry/vs_temperature/sln_viscosity_arrhenius_parameters.txt',
                            A_sigma=26.1, B_sigma=(-0.15), DeltaT = 20,
                            folder_for_frames='E:/levitating_droplet_with_heating/frames_map/',
                            shear_thinning=False):
    # load list of temperatures for all frames
    temperatures = np.loadtxt(temperatures_txt_filepath,
                              delimiter=',', skiprows=1)
    A, B = np.loadtxt(viscosity_arrhenius_parameters_file)

    if not shear_thinning:
        def viscosity_at_temperature(T):
            return A * np.exp(B/(T + 273.15)) / 1000
    else:
        lowshear_data = np.loadtxt('misc_data/rheometry/vs_temperature/sln_viscosity-vs-t_lowshear.csv', delimiter='\t',
                          skiprows=2, usecols=[6, 3])
        lowshear_interpolator = interp1d(lowshear_data[:, 0], lowshear_data[:, 1], kind='linear',
                                           fill_value='extrapolate')
        highshear_data = np.loadtxt('misc_data/rheometry/vs_temperature/sln_viscosity-vs-t_highshear.csv', delimiter='\t',
                          skiprows=2, usecols=[6, 3])
        highshear_interpolator = interp1d(highshear_data[:, 0], highshear_data[:, 1], kind='linear',
                                             fill_value='extrapolate')

        def viscosity_at_temperature(T, shear_rate):
            if shear_rate == 100:
                return highshear_interpolator(T) / 1000
            elif shear_rate == 1:
                return lowshear_interpolator(T) / 1000
            else:
                raise ValueError('Shear rate must be 1 or 100')

    def surface_tension_at_temperature(T):
        return A_sigma + B_sigma * (T - DeltaT)


    nframes = len(temperatures)
    print(nframes)
    # Plot of the parameters in the stability space
    size_factor = 0.90
    fig, ax = plt.subplots(dpi=300, figsize=(4.5*size_factor, 3.1*size_factor))
    # ax.plot(stability_boundary[:, 0], stability_boundary[:, 1], '--', color='grey', linewidth=1.5, alpha=0.8)
    # ax.fill_betweenx(x1=1e-5, x2=stability_boundary[:, 0], y=stability_boundary[:, 1], color='goldenrod', alpha=0.2)
    # ax.fill_betweenx(x1=1e3, x2=stability_boundary[:, 0], y=stability_boundary[:, 1], color='C0', alpha=0.2)
    ax.axvspan(xmin=1e-5, xmax=threshold_viscosity, color='goldenrod', alpha=0.2)
    ax.axvspan(xmax=1e3, xmin=threshold_viscosity, color='C0', alpha=0.2)
    plt.annotate('Stable\nlevitation', color='C0', xy=(0.40, 50), fontsize=16, alpha=0.6, ha='center')
    plt.annotate('No\nlevitation', color='goldenrod', xy=(0.0013, 50), fontsize=14, alpha=0.6, ha='center')
    ax.axvline(x=threshold_viscosity, linestyle='--', color='grey', linewidth=1.5, alpha=0.8)
    H = plt.scatter([50], [1], marker='o', color='C2')
    if shear_thinning:
        shearline, = plt.plot([50, 50], [1, 1], color='C2', linewidth=1)
    plt.xscale('log')
    plt.xlim(0.0002, 15)
    plt.ylim(10, 77)
    plt.xlabel('Viscosity $\mu$, Pa·s')
    plt.ylabel('Surface tension $\sigma$, mN·m$^{-1}$')
    plt.tight_layout()

    for frame_id in tqdm(range(nframes)):
        # if (frame_id < 3200) or (frame_id > 3300):
        #     continue
        temperature_here = temperatures[frame_id]
        surface_tension = surface_tension_at_temperature(temperature_here)

        if not shear_thinning:
            viscosity = viscosity_at_temperature(temperature_here)
            H.set_offsets(np.c_[[viscosity], [surface_tension]])
        else:
            viscosity_1 = viscosity_at_temperature(temperature_here, shear_rate=1)
            viscosity_100 = viscosity_at_temperature(temperature_here, shear_rate=100)
            H.set_offsets(np.c_[[viscosity_1], [surface_tension]])
            shearline.set_xdata([viscosity_1, viscosity_100])
            shearline.set_ydata([surface_tension, surface_tension])
        if frame_id % 6 < 3:
            H.set_alpha(1.0)
        else:
            H.set_alpha(0.3)

        # if is_stable(viscosity, surface_tension):
        #     H.set_color('C2')
        # else:
        #     H.set_color('C3')
        # H.set_alpha(0.9*(frame_id % 2) + 0.1)
        fig.savefig(f'{folder_for_frames}{frame_id:05d}.png', dpi=300)


def draw_tdeps_for_all_frames(temperatures_txt_filepath='E:/levitating_droplet_with_heating/ir-sensor-videos/ir_sensor_temperatures.txt',
                            viscosity_arrhenius_parameters_file='misc_data/rheometry/vs_temperature/sln_viscosity_arrhenius_parameters.txt',
                            A_sigma=26.1, B_sigma=(-0.15), DeltaT = 20,
                            folder_for_frames='E:/levitating_droplet_with_heating/frames_map/',
                            shear_thinning=False):
    # load list of temperatures for all frames
    temperatures = np.loadtxt(temperatures_txt_filepath,
                              delimiter=',', skiprows=1)
    A, B = np.loadtxt(viscosity_arrhenius_parameters_file)

    if not shear_thinning:
        def viscosity_at_temperature(T):
            return A * np.exp(B/(T + 273.15)) / 1000
    else:
        lowshear_data = np.loadtxt('misc_data/rheometry/vs_temperature/sln_viscosity-vs-t_lowshear.csv', delimiter='\t',
                          skiprows=2, usecols=[6, 3])
        lowshear_interpolator = interp1d(lowshear_data[:, 0], lowshear_data[:, 1], kind='linear',
                                           fill_value='extrapolate')
        highshear_data = np.loadtxt('misc_data/rheometry/vs_temperature/sln_viscosity-vs-t_highshear.csv', delimiter='\t',
                          skiprows=2, usecols=[6, 3])
        highshear_interpolator = interp1d(highshear_data[:, 0], highshear_data[:, 1], kind='linear',
                                             fill_value='extrapolate')

        def viscosity_at_temperature(T, shear_rate=1):
            if shear_rate == 100:
                return highshear_interpolator(T) / 1000
            elif shear_rate == 1:
                return lowshear_interpolator(T) / 1000
            else:
                raise ValueError('Shear rate must be 1 or 100')

    def surface_tension_at_temperature(T):
        return A_sigma + B_sigma * (T - DeltaT)


    nframes = len(temperatures)
    print(nframes)
    # Plot of the parameters in the stability space
    size_factor = 0.90
    fig, ax = plt.subplots(dpi=300, figsize=(4.5*size_factor, 2.6*size_factor))
    temps = np.linspace(20, 95, 100)
    ax.plot(temps, viscosity_at_temperature(temps, shear_rate=1), color='black', linewidth=1.5, alpha=0.8)
    ax.plot(temps, viscosity_at_temperature(temps, shear_rate=100), color='black', linewidth=1.5, alpha=0.3)
    viscdot = plt.scatter([20], [1], marker='o', color='black')
    # plt.yscale('log')
    ax.set_ylim(0, viscosity_at_temperature(20)*1.2)
    ax.set_xlim(20, 100)
    ax.set_xlabel('Temperature, °C')

    plt.ylabel('Viscosity $\mu$, Pa·s')

    ax2 = ax.twinx()
    ax2.set_ylim(0, surface_tension_at_temperature(20)*1.1)
    ax2.plot(temps, surface_tension_at_temperature(temps), color='C1', linewidth=1.5, alpha=0.8)
    ax2.yaxis.label.set_color('C1')
    ax2.spines["right"].set_edgecolor('C1')
    ax2.tick_params(axis='y', colors='C1')
    sigmadot = ax2.scatter([20], [40], marker='o', color='C1')

    vline = ax.axvline(20, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.ylabel('Surface tension $\sigma$, mN·m$^{-1}$')
    plt.tight_layout()
    # plt.show()

    for frame_id in tqdm(range(nframes)):
        # if (frame_id < 3200) or (frame_id > 3300):
        #     continue
        temperature_here = temperatures[frame_id]
        surface_tension = surface_tension_at_temperature(temperature_here)
        viscosity = viscosity_at_temperature(temperature_here)

        viscdot.set_offsets(np.c_[[temperature_here], [viscosity]])
        sigmadot.set_offsets(np.c_[[temperature_here], [surface_tension]])
        vline.set_xdata([temperature_here])
        if frame_id % 6 < 3:
            viscdot.set_alpha(1.0)
            sigmadot.set_alpha(1.0)
        else:
            viscdot.set_alpha(0.3)
            sigmadot.set_alpha(0.3)

        fig.savefig(f'{folder_for_frames}{frame_id:05d}.png', dpi=300)

# plt.show()

if __name__ == '__main__':
    # draw_map_for_all_frames(
    #     temperatures_txt_filepath='E:/levitating_droplet_with_heating/ir-sensor-videos/ir_sensor_temperatures.txt',
    #     viscosity_arrhenius_parameters_file='misc_data/rheometry/vs_temperature/sln_viscosity_arrhenius_parameters.txt',
    #     A_sigma=26.1, B_sigma=(-0.15), DeltaT=20,
    #     folder_for_frames='E:/levitating_droplet_with_heating/frames_map/',
    #     shear_thinning=True)
    #
    draw_map_for_all_frames(
        temperatures_txt_filepath='E:/levitating_droplet_with_heating_cyclohexanol/ir_sensor_videos/ir_sensor_temperatures.txt',
        viscosity_arrhenius_parameters_file='misc_data/rheometry/vs_temperature/cyclohexanol_viscosity_arrhenius_parameters.txt',
        A_sigma=3.468324607301324392e+01,
        B_sigma=-7.816005982730422907e-02,
        DeltaT=0,
        folder_for_frames='E:/levitating_droplet_with_heating_cyclohexanol/frames_map/')
    #
    # draw_tdeps_for_all_frames(
    #     temperatures_txt_filepath='E:/levitating_droplet_with_heating/ir-sensor-videos/ir_sensor_temperatures.txt',
    #     viscosity_arrhenius_parameters_file='misc_data/rheometry/vs_temperature/sln_viscosity_arrhenius_parameters.txt',
    #     A_sigma=26.1, B_sigma=(-0.15), DeltaT=20,
    #     folder_for_frames='E:/levitating_droplet_with_heating/tdeps_map/',
    #     shear_thinning=True)

    # draw_tdeps_for_all_frames(
    #     temperatures_txt_filepath='E:/levitating_droplet_with_heating_cyclohexanol/ir_sensor_videos/ir_sensor_temperatures.txt',
    #     viscosity_arrhenius_parameters_file='misc_data/rheometry/vs_temperature/cyclohexanol_viscosity_arrhenius_parameters.txt',
    #     A_sigma=3.468324607301324392e+01,
    #     B_sigma=-7.816005982730422907e-02,
    #     DeltaT=0,
    #     folder_for_frames='E:/levitating_droplet_with_heating_cyclohexanol/tdeps_map/')