import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage import io
from scipy import ndimage as ndi
from skimage.filters import gaussian
import pickle
from scipy.signal import find_peaks
import os
import time
from glob import glob
import configparser

# target_folder = 'F:/yankai_levitation_transfer_highspeed_videos/10Hz_300V_sinewave_1/whole_video_1/'
# raw_frames_suffix = 'FASTS4_000000/'

def process_one_folder(target_folder,
                        start_frame = 0,
                        sum_0 = 1,
                        fps=False):
    video_folder = target_folder
    filelist = []
    # find all "imgNNN" subfolders
    img_subfolders = sorted(glob(video_folder + "img*/"))
    for video_subfolder in img_subfolders:
        for file in sorted(os.listdir(video_subfolder)):
            if file.endswith(".jpg"):
                filelist.append(os.path.join(video_subfolder, file))
    print('Found {0} jpg frames.'.format(len(filelist)))
    frame_id_max = len(filelist) - 1

    config = configparser.ConfigParser()
    config.read(video_folder + 'metadata.txt')
    if not fps:
        fps = int(config['Record']['fps'])
    print('FPS is: {0}'.format(fps))

    def load_image(frame_id):
        # return np.sum(io.imread('{0}{1:07d}.jpg'.format(video_folder, frame_id)).astype(float), axis=2)
        return np.sum(io.imread(filelist[frame_id]).astype(float), axis=2)

    def load_normed(frame_id):
        image = load_image(frame_id)
        return image*sum_0

    t0 = time.time()
    mean_intensity = []
    for frame_id in range(start_frame, frame_id_max):
        print(frame_id)
        image = load_normed(frame_id)
        mean_intensity.append(np.sum(image))
        if frame_id % 1000 == 0:
            np.save('misc_data/fps_calibration_trace', np.array(mean_intensity))

    np.save('fps_calibration_trace', np.array(mean_intensity))
    print('Speed was {0:.2f} frames/min'.format((frame_id_max - start_frame)/(time.time()-t0)*60))
    plt.show()

if __name__ == '__main__':
    # process_one_folder('E:/LED flashing/LED flashing/FASTS4_000000/',
    #                    start_frame=0)
    x = np.load('misc_data/fps_calibration_trace.npy')
    peaks, _ = find_peaks(x, distance=25)
    # plt.plot(x)
    # plt.plot(peaks, x[peaks], "x")
    # plt.show()
    print(peaks[0])
    print(peaks[-1])
    total_frames = (peaks[-1]-peaks[0])
    total_time = (len(peaks)-1)/1000 # in seconds
    print(total_frames/total_time)

    # process_one_folder('E:/2020_nov_30/01_FASTS4_cleardrop_reg_good2_160x100pics_000000/',
    #                    scope_filename='scope_1-30clear130.csv',
    #                    start_frame=0,
    #                    center=(38, 126), size=8)
    # target_folder = 'E:/2020_nov_30/00_FASTS4_cleardrop_reg_160x60pics_good2_000000/'
    # scope_traces = np.loadtxt(target_folder+'scope_1-30clear132.csv', delimiter=',', skiprows=24)
    # scope_time = scope_traces[:,0]
    # scope_trigger = scope_traces[:,1]
    # scope_extv = scope_traces[:, 2]
    # scope_droplet = scope_traces[:, 3]
    #
    # plt.plot(scope_traces[:,0], scope_traces[:,1])
    # plt.plot(scope_traces[:, 0], scope_traces[:, 2])
    # plt.plot(scope_traces[:, 0], scope_traces[:, 3])
    #
    # camera_start_id = np.argmax(scope_trigger > 1.5)
    # camera_start_time = scope_time[camera_start_id]
    #
    # plt.show()