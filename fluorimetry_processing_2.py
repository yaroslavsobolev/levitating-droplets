import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog
from scipy.signal import savgol_filter
import re
import pickle

def get_header(ref_file, skip_header=5):
    headers = []
    with open(ref_file) as f:
        for i in range(skip_header):
            header = f.readline()
        # print(header.replace('\t', '\r\n'))
        headers = header.split('\t')
    return headers

def get_coeff(main_reference_spectrum, target_spectrum, target_spectrum_wavelengths, slope, intercept, noise_std, reference_wavelengths,
              only_rh_reference=True, baseline=False, second_baseline_wavelengths=False, second_baseline=False):
    # def func(xs,a):
    #     res = []
    #     for x in xs:
    #         itemindex = np.where(wavelengths==x)[0][0]
    #         res.append(a*main_reference_spectrum[itemindex])
    #     return res
    if only_rh_reference:
        reference_interpolator = interpolate.interp1d(reference_wavelengths, main_reference_spectrum, fill_value='extrapolate')
        def func(xs, a):
            return a*reference_interpolator(xs)
        p0 = slope
        bounds = (-np.inf, np.inf)
    else:
        reference_interpolator = interpolate.interp1d(reference_wavelengths, main_reference_spectrum, fill_value='extrapolate')
        baseline_interpolator = interpolate.interp1d(reference_wavelengths, baseline,
                                                      fill_value='extrapolate')
        second_baseline_interpolator = interpolate.interp1d(second_baseline_wavelengths, second_baseline,
                                                      fill_value='extrapolate')
        def func(xs, a, b, c):
            return a*reference_interpolator(xs) + b*baseline_interpolator(xs) + c*second_baseline_interpolator(xs)
        p0 = (slope, 0.5, 0.5)
        bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    popt, pcov = curve_fit(func, target_spectrum_wavelengths, target_spectrum,
                           p0=p0, bounds=bounds,
                           sigma=noise_std*np.ones_like(target_spectrum),
                           absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    slope = popt[0]
    slope_error = perr[0]

    if not only_rh_reference:
        plt.plot(target_spectrum_wavelengths, target_spectrum, 'o', markersize=2, label='target spectrum')
        plt.plot(target_spectrum_wavelengths, target_spectrum-second_baseline_interpolator(target_spectrum_wavelengths),
                 label='target spectrum minus 2nd baseline')
        plt.plot(target_spectrum_wavelengths, func(target_spectrum_wavelengths, popt[0], popt[1], popt[2]),
                 label='fit spectrum', alpha=0.8)
        plt.plot(target_spectrum_wavelengths, func(target_spectrum_wavelengths, popt[0], 0, 0),
                 label='RhB component')
        plt.plot(target_spectrum_wavelengths, func(target_spectrum_wavelengths, 0, popt[1], 0),
                 label='1st baseline component')
        plt.plot(target_spectrum_wavelengths, func(target_spectrum_wavelengths, 0, 0, popt[2]),
                 label='2nd baseline component')
        # plt.plot(target_spectrum_wavelengths, func(target_spectrum_wavelengths, 0, 0, 0, popt[3]),
        #          label='support')

    return slope, slope_error

def construct_reference(ref_file, ax):
    # NOTES ABOUT REFERENCE SPECTRA:
    # Concentrations
    # ORI = detergent(including 0.25%Rh.B w/w)+MeOH=1:5 w/w
    # Base = detergent+MeOH 1:5 w/w
    # ORIE2 = ORI dilutied 100 times
    # ORIE3 = ORI dilutied 1000 times
    # =================================
    # index samplename
    # 0 	ORIE5-quartz_cell-D1 540:550-700
    # 1	    T637375131720992419
    # 2 	ORIE5-polystyrene_cell-1st measurement-D1 540:550-700
    # 3 	T637375131720992864
    # 4 	ORIE5-polystyrene_cell-2nd measurement-D1 540:550-700
    # 5 	T637375131720993309
    # 6 	ORIE6-1st_measurement-D1 540:550-700
    # 7 	T637375131720993754
    # 8	    ORIE6-2nd_measurement-D1 540:550-700
    # 9	    T637375131720994199
    # 10	ORIE4-1st_measurement-D1 540:550-700
    # 11	T637375131720994644
    # 12	ORIE4-2nd_measurement-D1 540:550-700
    # 13	T637375131720995089
    # 14	ORIE3-1st_measurement-D1 540:550-700
    # 15	T637375131720995534
    # 16	ORIE3-2nd_measurement-D1 540:550-700
    # 17	T637375131720995979
    # 18	ORIE7-1st_measurement-D1 540:550-700
    # 19	T637375131720996424
    # 20	ORIE7-2nd_measurement-D1 540:550-700
    # 21	T637375131720996869
    # 22	Base_line-2nd measurement-D1 540:550-700
    # 23	T637375131720997314
    # 24	Base_line-1st measurement-D1 540:550-700
    # 25	T637375131720997759
    # 26	ORIE2-1st_measurement-D1 540:550-700
    # 27	T637375131720998204
    # 28	ORIE2-2nd_measurement-D1 540:550-700
    # 29	T637375131720998649
    ref_data = np.genfromtxt(ref_file, skip_header=6, skip_footer=2, delimiter='\t')
    headers = get_header(ref_file=ref_file)
    ref_spectra = []
    wavelengths = ref_data[:,0]
    for col_id in range(0, ref_data.shape[1]-1, 2):
        assert (ref_data[:,col_id] == wavelengths).all()
        label = headers[col_id]
        if label.find('quartz') > 0:
            concentration = -1
        elif label[:3] == 'sol':
            concentration = 0
        else:
            concentration = 10**(-1*int(label[4]))
        ref_spectra.append([concentration, label, ref_data[:,col_id+1]])
        # plt.plot(wavelengths, ref_data[:,col_id+1], label=headers[col_id])

    baseline = np.mean([r[2] for r in ref_spectra if r[0] == 0], axis=0)
    noise_std = np.std((ref_spectra[9][2] - baseline)[-20:])

    unique_concentrations = sorted(list(set([r[0] for r in ref_spectra])))
    calib_rh = {concentration : np.mean([r[2] for r in ref_spectra if r[0] == concentration], axis=0) - baseline
                for concentration in unique_concentrations if (concentration > 0)}
    main_reference_spectrum = calib_rh[1E-2]
    calibration_slopes = [[0, 0, 0]]
    for c in calib_rh:
        y = calib_rh[c]
        slope, intercept, r_value, p_value, std_err = stats.linregress(main_reference_spectrum, y)
        slope, _ = get_coeff(main_reference_spectrum, y, wavelengths, slope, intercept, noise_std,
                                       wavelengths)
        calibration_slopes.append([c, slope, intercept])
    calibration_slopes = np.array(calibration_slopes)
    xs = np.logspace(-4, 1, 100)
    slope_to_concentration_converter = interpolate.interp1d(calibration_slopes[:,1], calibration_slopes[:,0], fill_value='extrapolate')
    ax.loglog(calibration_slopes[:,0], calibration_slopes[:,1], 'o', label='Calibration points')
    ax.loglog(slope_to_concentration_converter(xs), xs, label='interpolator')
    # plt.show()
    return wavelengths, baseline, noise_std, main_reference_spectrum, slope_to_concentration_converter

def get_concentration_from_spectrum(spectrum_file, spectrum_id=1,
        ref_file='fluorescence_data/reference_rhodamine/2020_oct_08/For_ref_Acquisition 1 2020-10-07 13_58_43 »» Detector1.group.txt',
        second_baseline_file='fluorescence_data/reference_rhodamine/second_baseline.txt'):
    f_calib, ax_calib = plt.subplots()
    reference_wavelengths, baseline, noise_std, main_reference_spectrum, slope_to_concentration_converter = construct_reference(ref_file,
                                                                                                                                ax_calib)
    spectrum_id = 0 + 2*(spectrum_id-1)
    file_data = np.genfromtxt(spectrum_file, skip_header=6, skip_footer=2, delimiter='\t')
    target_spectrum_wavelengths = file_data[:, 0]
    # assert (file_data[:, 0] == wavelengths).all()
    header = get_header(spectrum_file, skip_header=5)
    print('Using trace with label {0}'.format(header[spectrum_id]))
    ax_calib.set_title(header[spectrum_id])
    # resample baseline into new wavelength range
    # baseline_interp = interpolate.interp1d(reference_wavelengths, baseline, fill_value='extrapolate')
    # resampled_baseline = baseline_interp(target_spectrum_wavelengths)
    target_spectrum = file_data[:, spectrum_id + 1]# - resampled_baseline
    # fig5 = plt.figure(5)
    # plt.plot(target_spectrum)
    # plt.plot(savgol_filter(target_spectrum, 31, 3))
    # plt.show()
    print('Baseline noise was: {0:.2f}'.format(noise_std))
    noise_std = np.std((target_spectrum - savgol_filter(target_spectrum, 31, 3))[-20:])
    print('Noise in current target spectrum is: {0:.2f}'.format(noise_std))

    second_baseline_data = np.genfromtxt(second_baseline_file, skip_header=4, skip_footer=2, delimiter='\t')
    # slope, intercept, r_value, p_value, std_err = stats.linregress(main_reference_spectrum, target_spectrum)
    fig9 = plt.figure(9)
    slope, slope_error = get_coeff(main_reference_spectrum, target_spectrum, target_spectrum_wavelengths,
                                   slope=1e-3, intercept=0, noise_std=noise_std,
                                   reference_wavelengths=reference_wavelengths,
                                   baseline=baseline, only_rh_reference=False,
                                   second_baseline_wavelengths=second_baseline_data[:, 0],
                                   second_baseline=second_baseline_data[:,1])
    plt.legend()
    # plt.show()
    concentration = slope_to_concentration_converter(slope)
    conc_error = slope_to_concentration_converter(slope + slope_error) - concentration
    ax_calib.plot(concentration, slope, 'o', label='target spectrum')
    ax_calib.set_xlabel('Dilution from ORI')
    ax_calib.set_ylabel('Scaling factor with respect to dilution 1E-2')
    ax_calib.legend()
    f5 = plt.figure(5)
    plt.semilogy(reference_wavelengths, main_reference_spectrum, 'o-', label='reference 1e-2')
    plt.semilogy(target_spectrum_wavelengths, target_spectrum, 'o-', label='Target spectrum')
    plt.legend()
    # f5 = plt.figure(6)
    # plt.plot(reference_wavelengths, main_reference_spectrum, 'o-', label='reference 1e-3')
    # plt.plot(target_spectrum_wavelengths, target_spectrum, 'o-', label='Target spectrum (baseline subtracted)')
    # plt.plot(target_spectrum_wavelengths, file_data[:, spectrum_id + 1], 'o-', label='Target spectrum raw')
    # plt.plot(reference_wavelengths, slope*main_reference_spectrum, 'o-', label='Fit of RH spectrum')
    # plt.legend()
    return concentration, conc_error

def get_transfer_volume(target_file, net_volume_in_mL, number_of_droplets, spectrum_id=1):
    dilution, dilution_err = get_concentration_from_spectrum(target_file, spectrum_id=spectrum_id)
    vol_per_droplet = dilution*net_volume_in_mL*1e-3/number_of_droplets
    print('Dilution is {0}\n'
          'Volume per droplet is {1:.3f} pL\n'
          'Relative standard error is {2:.2f}%\n'
          '(This includes the random error only. The systematic error is not included.)'.format(dilution,
                                                       vol_per_droplet/1e-12,
                                                       100*dilution_err/dilution))
    print('Diameter of the droplet printed is: {0:.3f} microns'.format(2*(vol_per_droplet*3/4/np.pi)**(1/3)*1e5))
    plt.show()

def process_one_spectrum_with_auto_parameters(target_file, spectrum_id, net_volume_in_mL):
    header = get_header(target_file, skip_header=5)[0 + 2*(spectrum_id-1)]
    regexp1 = re.compile('''(?P<date>.+?)_(?P<label>\d+?)_(?P<vpp>.+?)vpp_(?P<rpm>.+?)rpm_(?P<cycles>.+?)cyc_(?P<id>.+?)$''',
                         re.DOTALL)
    match = regexp1.match(header)
    parameters = {s : match.group(s) for s in ['date', 'label', 'vpp', 'rpm', 'cycles', 'id']}
    for s in ['vpp', 'rpm', 'cycles']:
        parameters[s] = float(parameters[s])
    dilution, dilution_err = get_concentration_from_spectrum(target_file, spectrum_id=spectrum_id)
    plt.close('all')
    vol_per_droplet = dilution*net_volume_in_mL*1e-3/(2*parameters['cycles'])
    droplet_diameter = 2 * (vol_per_droplet * 3 / 4 / np.pi) ** (1 / 3) * 1e5
    rel_error = 100*dilution_err/dilution
    parameters['vol_per_droplet'] = vol_per_droplet/1e-12
    parameters['droplet_diameter'] = droplet_diameter
    parameters['rel_error'] = rel_error
    return parameters

def process_all_experiments_in_file(target_file, net_volume_in_mL):
    results = []
    headers = get_header(target_file, skip_header=5)
    last_spectrum_id = int(round(len(headers)/2+1))
    for spectrum_id in range(last_spectrum_id):
        print('\n')
        results.append(process_one_spectrum_with_auto_parameters(target_file, spectrum_id, net_volume_in_mL))
    return results

def save_results_to_files(results, filename_for_saving):
    with open(filename_for_saving + '.txt', 'w+') as f:
        s = ''
        cols = results[0].keys()
        for c in cols:
            s += c + '\t'
        f.write(s[:-1] + '\n')
        for record in results:
            s = ''
            for c in cols:
                s += '{0}\t'.format(record[c])
            f.write(s[:-1] + '\n')
    pickle.dump(results, open(filename_for_saving + '.pickle', "wb"))

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    target_file = filedialog.askopenfilename()
    results = process_all_experiments_in_file(target_file, net_volume_in_mL=2.05/0.8)
    save_results_to_files(results, target_file[:-3]+'_processed')

    ## This is for processing a single trace and showing all the plots, etc.
    # root = tk.Tk()
    # root.withdraw()
    # target_file = filedialog.askopenfilename()
    # get_transfer_volume(target_file=target_file,
    #                     spectrum_id=1,
    #                     net_volume_in_mL=2.05/0.8,  #The net volume in the drum (mL)
    #                     number_of_droplets=400) #Number of material transfer events