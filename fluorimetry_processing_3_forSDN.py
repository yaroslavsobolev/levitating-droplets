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
        # second_baseline_interpolator = interpolate.interp1d(second_baseline_wavelengths, second_baseline,
        #                                               fill_value='extrapolate')
        def func(xs, a, b):
            return a*reference_interpolator(xs) + b*baseline_interpolator(xs)# + c*second_baseline_interpolator(xs)
        p0 = (slope, 0.5)#, 0.5)
        # p0 = (slope, 0.01, 0.01)
        bounds = ([0, -np.inf], [np.inf, np.inf])
    popt, pcov = curve_fit(func, target_spectrum_wavelengths, target_spectrum,
                           p0=p0, bounds=bounds,
                           sigma=noise_std*np.ones_like(target_spectrum),
                           absolute_sigma=True)
                           # ftol=1.49012e-09, xtol=1.49012e-09)
    perr = np.sqrt(np.diag(pcov))
    slope = popt[0]
    slope_error = perr[0]

    if not only_rh_reference:
        plt.title('RhB,1st_baseline,2nd_baseline={0}'.format(popt))
        plt.plot(target_spectrum_wavelengths, target_spectrum, 'o', markersize=2, label='target spectrum')
        # plt.plot(target_spectrum_wavelengths, target_spectrum-second_baseline_interpolator(target_spectrum_wavelengths),
        #          label='target spectrum minus 2nd baseline', alpha=0.3)
        # plt.plot(target_spectrum_wavelengths, target_spectrum-baseline_interpolator(target_spectrum_wavelengths),
        #          label='target spectrum minus 1st baseline', alpha=0.3)
        plt.plot(target_spectrum_wavelengths, func(target_spectrum_wavelengths, popt[0], popt[1]),
                 label='fit spectrum', alpha=0.8)
        plt.plot(target_spectrum_wavelengths, func(target_spectrum_wavelengths, popt[0], 0),
                 label='RhB component')
        plt.plot(target_spectrum_wavelengths, func(target_spectrum_wavelengths, 0, popt[1]),
                 label='1st baseline component')
        # plt.plot(target_spectrum_wavelengths, func(target_spectrum_wavelengths, 0, 0, popt[2]),
        #          label='2nd baseline component')
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
        if label.find('baseline_Second') >= 0:
            concentration = -1
        elif label.find('Baseline_new') >= 0:
            concentration = 0
        else:
            concentration = 10**(-1*int(label[4]))
        if concentration == 1e-4:
            continue
        ref_spectra.append([concentration, label, ref_data[:,col_id+1]])
        # plt.plot(wavelengths, ref_data[:,col_id+1], label=headers[col_id])

    baseline = np.mean([r[2] for r in ref_spectra if r[0] == 0], axis=0)
    noise_std = np.std((ref_spectra[2][2] - baseline)[-20:])

    unique_concentrations = sorted(list(set([r[0] for r in ref_spectra])))
    calib_rh = {concentration : np.mean([r[2] for r in ref_spectra if r[0] == concentration], axis=0) - baseline
                for concentration in unique_concentrations if (concentration > 0)}
    main_reference_spectrum = calib_rh[1E-5]
    calibration_slopes = [[0, 0, 0]]
    for c in calib_rh:
        y = calib_rh[c]
        slope, intercept, r_value, p_value, std_err = stats.linregress(main_reference_spectrum, y)
        slope, _ = get_coeff(main_reference_spectrum, y, wavelengths, slope, intercept, noise_std,
                                       wavelengths)
        calibration_slopes.append([c, slope, intercept])
    calibration_slopes = np.array(calibration_slopes)
    xs = np.logspace(-9, 1, 100)
    slope_to_concentration_converter = interpolate.interp1d(calibration_slopes[:,1], calibration_slopes[:,0], fill_value='extrapolate')
    ax.loglog(calibration_slopes[:,0], calibration_slopes[:,1], 'o', label='Calibration points')
    ax.loglog(slope_to_concentration_converter(xs), xs, label='interpolator')
    # plt.show()
    return wavelengths, baseline, noise_std, main_reference_spectrum, slope_to_concentration_converter

def construct_reference_enhanced(ref_file, ax):
    # "ehnanced" means that the fluorimeter has the following settings:
    # Excitation monochromator: entrance slit 8 nm, exit slit 8 nm
    # Emission monochromator: entrance slit 4 nm, exit slit 4 nm
    # NOTES ABOUT REFERENCE SPECTRA header:
    #
    # ORI = detergent(including 0.25%Rh.B w/w)+MeOH=1:5 w/w
    # Base = detergent+MeOH 1:5 w/w
    # ORIE2 = ORI dilutied 100 times
    # ORIE3 = ORI dilutied 1000 times
    # =================================
    ref_data = np.genfromtxt(ref_file, skip_header=6, skip_footer=2, delimiter='\t')
    headers = get_header(ref_file=ref_file)
    ref_spectra = []
    wavelengths = ref_data[:,0]
    for col_id in range(0, 14, 2):
        assert (ref_data[:,col_id] == wavelengths).all()
        label = headers[col_id]
        if not (label.find('enhanced signal') >= 0):
            print('no enhanced signal')
            concentration = -1
        elif label.find('baseline') >= 0:
            concentration = 0
        elif label.find('ORIE') >= 0:
            concentration = 10**(-1*int(label[4]))
        else:
            continue
        ref_spectra.append([concentration, label, ref_data[:,col_id+1]])
        # plt.plot(wavelengths, ref_data[:,col_id+1], label=headers[col_id])

    baseline = np.mean([r[2] for r in ref_spectra if r[0] == 0], axis=0)
    noise_std = np.std((ref_spectra[3][2] - baseline)[-20:])

    unique_concentrations = sorted(list(set([r[0] for r in ref_spectra])))
    calib_rh = {concentration : np.mean([r[2] for r in ref_spectra if r[0] == concentration], axis=0) - baseline
                for concentration in unique_concentrations if (concentration > 0)}
    main_reference_spectrum = calib_rh[1E-4]
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
        # ref_file='fluorescence_data/2021_may_18/NewREF_Acquisition 1 2021-05-18 02_00_16.session.txt',
        ref_file='D:\\Dropbox_Folder\\Dropbox\\python_projects\\levitating-droplets\\fluorescence_data\\reference_rhodamine\\2021_may_18_NewREF_forSDN\\NewREF_Acquisition 1 2021-05-18 02_00_16.session__from558nm_2.txt',
        enhanced_signal_settings=False):
    f_calib, ax_calib = plt.subplots()
    if enhanced_signal_settings:
        # second_baseline_file = 'fluorescence_data/2021_may_16_new_reference/GOOD_Acquisition 1 2021-05-16 22_27_10 »» Detector1.group.txt'
        # second_baseline_data = np.genfromtxt(second_baseline_file, skip_header=6, skip_footer=2, delimiter='\t')
        reference_wavelengths, baseline, noise_std, main_reference_spectrum, slope_to_concentration_converter = \
            construct_reference_enhanced(
                # 'fluorescence_data/2021_may_18/NewREF_Acquisition 1 2021-05-18 02_00_16.session.txt',
                '/fluorescence_data/2021_may_18/NewREF_Acquisition 1 2021-05-18 02_00_16.session__from558nm.txt',
                ax_calib)
    else:
        # second_baseline_file = 'fluorescence_data/reference_rhodamine/second_baseline.txt'
        # second_baseline_data = np.genfromtxt(second_baseline_file, skip_header=4, skip_footer=2, delimiter='\t')
        reference_wavelengths, baseline, noise_std, main_reference_spectrum, slope_to_concentration_converter = construct_reference(ref_file,
                                                                                                                                ax_calib)
    spectrum_id = 0 + 2*(spectrum_id-1)
    file_data = np.genfromtxt(spectrum_file, skip_header=6, skip_footer=2, delimiter='\t')
    target_spectrum_wavelengths = file_data[:, spectrum_id]
    target_spectrum_wavelengths = target_spectrum_wavelengths[~np.isnan(target_spectrum_wavelengths)]
    # assert (file_data[:, 0] == wavelengths).all()
    header = get_header(spectrum_file, skip_header=5)
    print('Using trace with label {0}'.format(header[spectrum_id]))
    ax_calib.set_title(header[spectrum_id])
    # resample baseline into new wavelength range
    # baseline_interp = interpolate.interp1d(reference_wavelengths, baseline, fill_value='extrapolate')
    # resampled_baseline = baseline_interp(target_spectrum_wavelengths)
    target_spectrum = file_data[:, spectrum_id + 1]# - resampled_baseline
    target_spectrum = target_spectrum[~np.isnan(target_spectrum)]
    if enhanced_signal_settings:
        cut_from_the_start = 4
        target_spectrum = target_spectrum[cut_from_the_start:]
        target_spectrum_wavelengths = target_spectrum_wavelengths[cut_from_the_start:]
    # fig5 = plt.figure(5)
    # plt.plot(target_spectrum)
    # plt.plot(savgol_filter(target_spectrum, 31, 3))
    # plt.show()
    print('Baseline noise was: {0:.2f}'.format(noise_std))
    noise_std = np.std((target_spectrum - savgol_filter(target_spectrum, 31, 3))[-20:])
    print('Noise in current target spectrum is: {0:.2f}'.format(noise_std))

    # slope, intercept, r_value, p_value, std_err = stats.linregress(main_reference_spectrum, target_spectrum)
    fig9 = plt.figure(9)
    slope, slope_error = get_coeff(main_reference_spectrum, target_spectrum, target_spectrum_wavelengths,
                                   slope=1e-3, intercept=0, noise_std=noise_std,
                                   reference_wavelengths=reference_wavelengths,
                                   baseline=baseline, only_rh_reference=False,
                                   second_baseline_wavelengths=reference_wavelengths,
                                   second_baseline=baseline)
    plt.legend()
    #plt.show()                                        ##<<<<< UNCOMMENT THIS 'plt.show()' LINE TO SEE THE FIT QUALITY
    concentration = slope_to_concentration_converter(slope)
    conc_error = slope_to_concentration_converter(slope + slope_error) - concentration
    ax_calib.plot(concentration, slope, 'o', label='target spectrum')
    ax_calib.set_xlabel('Dilution from ORI')
    ax_calib.set_ylabel('Scaling factor with respect to dilution 1E-5')
    ax_calib.legend()
    f5 = plt.figure(5)
    plt.semilogy(reference_wavelengths, main_reference_spectrum, 'o-', label='reference 1e-5')
    plt.semilogy(target_spectrum_wavelengths, target_spectrum, 'o-', label='Target spectrum')
    plt.legend()
    # f5 = plt.figure(6)
    # plt.plot(reference_wavelengths, main_reference_spectrum, 'o-', label='reference 1e-3')
    # plt.plot(target_spectrum_wavelengths, target_spectrum, 'o-', label='Target spectrum')
    # # plt.plot(target_spectrum_wavelengths, file_data[4:, spectrum_id + 1], 'o-', label='Target spectrum raw')
    # plt.plot(reference_wavelengths, slope*main_reference_spectrum, 'o-', label='RH component')
    # plt.legend()
    # plt.show()
    return concentration, conc_error

def get_transfer_volume(target_file, net_volume_in_mL, number_of_droplets, spectrum_id=1,
                        enhanced_signal_settings=False):
    dilution, dilution_err = get_concentration_from_spectrum(target_file, spectrum_id=spectrum_id,
                                                             enhanced_signal_settings=enhanced_signal_settings)
    vol_per_droplet = dilution*net_volume_in_mL*1e9/number_of_droplets
    print('Dilution is {0}\n'
          'Volume per droplet is {1:.3f} pL\n'
          'Relative standard error is {2:.2f}%\n'
          '(This includes the random error only. The systematic error is not included.)'.format(dilution,
                                                       vol_per_droplet,
                                                       100*dilution_err/dilution))
    print('Diameter of the droplet printed is: {0:.3f} microns'.format(2*(vol_per_droplet*3/4/np.pi)**(1/3)*1e5))
    plt.show()

def process_one_spectrum_with_auto_parameters(target_file, spectrum_id, net_volume_in_mL,
                                              force_enhanced_signal_settings=False):
    header = get_header(target_file, skip_header=5)[0 + 2*(spectrum_id-1)]
    regexp1 = re.compile('''(?P<date>.+?)_(?P<label>\d+?)_(?P<vpp>.+?)vpp_(?P<rpm>.+?)rpm_(?P<cycles>.+?)cyc_(?P<id>.+?)$''',
                         re.DOTALL)
    match = regexp1.match(header)
    if not match:
        return False
    parameters = {s : match.group(s) for s in ['date', 'label', 'vpp', 'rpm', 'cycles', 'id']}
    for s in ['vpp', 'rpm', 'cycles']:
        parameters[s] = float(parameters[s])
    if force_enhanced_signal_settings:
        enhanced_signal_settings = True
    else:
        if parameters['id'].find('enhanced signal') >= 0:
            enhanced_signal_settings = True
        else:
            enhanced_signal_settings = False

    dilution, dilution_err = get_concentration_from_spectrum(target_file, spectrum_id=spectrum_id,
                                                             enhanced_signal_settings=enhanced_signal_settings)
    plt.close('all')
    vol_per_droplet = dilution*net_volume_in_mL*1e-3/(2*parameters['cycles'])
    droplet_diameter = 2 * (vol_per_droplet * 3 / 4 / np.pi) ** (1 / 3) * 1e5
    rel_error = 100*dilution_err/dilution
    parameters['vol_per_droplet'] = vol_per_droplet/1e-12
    parameters['droplet_diameter'] = droplet_diameter
    parameters['rel_error'] = rel_error
    return parameters

def process_all_experiments_in_file(target_file, net_volume_in_mL,
                                    force_enhanced_signal_settings=False):
    results = []
    headers = get_header(target_file, skip_header=5)
    last_spectrum_id = int(round(len(headers)/2+1))
    for spectrum_id in range(last_spectrum_id):
        print('\n')
        processing_result = process_one_spectrum_with_auto_parameters(target_file, spectrum_id, net_volume_in_mL,
                                                                 force_enhanced_signal_settings=force_enhanced_signal_settings)
        if processing_result:
            results.append(processing_result)
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
    ## THIS IS FOR TESTING THE SINGLE SPECTRUM PROCESSING
    # target_file = 'fluorescence_data/2020_oct/oct_27/Calibration for new setting/For_calibration_of_new_settings_oct_27.txt'
    # results = process_one_spectrum_with_auto_parameters(target_file, spectrum_id=1, net_volume_in_mL=2.05/0.8,
    #                                           force_enhanced_signal_settings=False)
    # print(results)

    # ## This is a simple interface asking for a file and processing all spectra in it
    # root = tk.Tk()
    # root.withdraw()
    # target_file = filedialog.askopenfilename()
    # results = process_all_experiments_in_file(target_file, net_volume_in_mL=2.05/0.8)
    # save_results_to_files(results, target_file[:-3]+'_processed')

    # This is for processing a single trace and showing all the plots, etc.
    root = tk.Tk()
    root.withdraw()
    target_file = filedialog.askopenfilename()
    get_transfer_volume(target_file=target_file,
                        spectrum_id=1,
                        net_volume_in_mL=12,  #The net volume in the drum (mL)
                        number_of_droplets=1000) #Number of material transfer events