import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage import io
from scipy import ndimage as ndi
from skimage.filters import gaussian
import pickle
from skimage.feature import register_translation
from skimage.feature import canny
from skimage.filters import unsharp_mask
from skimage import filters
import os
import time
from glob import glob
import configparser

# target_folder = 'F:/yankai_levitation_transfer_highspeed_videos/10Hz_300V_sinewave_1/whole_video_1/'
# raw_frames_suffix = 'FASTS4_000000/'

def process_one_folder(target_folder,
                        scope_filename,
                        thresh_1 = 15,
                        start_frame = 0,
                        do_canny=False,
                        do_figures = True,
                        low_threshold=30,
                        high_threshold=150,
                        radius=False,
                        time_scale=1,
                        center=False,
                        size=7,
                        sum_0 = 1,
                        dt = 0.002,
                        fps=False):
    video_folder = target_folder
    frames_folder = target_folder + 'frames_correlated/'
    for my_folder in [frames_folder]:
        if not os.path.exists(my_folder):
            os.makedirs(my_folder)
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

    plt.imshow(load_normed(1), cmap='plasma', vmax=256*3)
    plt.show()

    prev_image = load_normed(0)

    def get_median_bkg(from_frame_id=0, length=200):
        print('Getting median bkg')
        images = []
        for frame_id in range(from_frame_id, from_frame_id+length):
            images.append(load_normed(frame_id))
        images = np.array(images)
        im_median = np.median(images, axis=0)
        return im_median

    bkg_median = get_median_bkg(start_frame)
    
    def img_zoom(image):
        return image[center[0]-size : center[0]+size, center[1]-size : center[1]+size]
    # center=(200, 200)

    scope_traces = np.loadtxt(target_folder + scope_filename, delimiter=',', skiprows=24)
    scope_time = scope_traces[:,0]*time_scale
    scope_time -= scope_time[0]
    scope_trigger = scope_traces[:,1]
    scope_extv = scope_traces[:, 2]
    scope_droplet = scope_traces[:, 3]

    # plt.plot(scope_traces[:,0], scope_traces[:,1])
    # plt.plot(scope_traces[:, 0], scope_traces[:, 2])
    # plt.plot(scope_traces[:, 0], scope_traces[:, 3])

    camera_start_id = np.argmax(scope_trigger > 1.5)
    camera_start_time = scope_time[camera_start_id]
    const_delay = 0 #3.68471 - 3.6830
    def camera_frame_to_trace_time(frame_id):
        return frame_id/fps + camera_start_time - const_delay

    # plt.show()

    t0 = time.time()
    for frame_id in range(start_frame, frame_id_max):
        print(frame_id)
        prev_image = load_normed(frame_id-1)
        image = load_normed(frame_id)
        diffimg = image-prev_image

        fig3 = plt.figure(constrained_layout=True, figsize=(6,5))
        gs = fig3.add_gridspec(2, 2)
        ax_scope = fig3.add_subplot(gs[0, 0])
        ax_zoom = fig3.add_subplot(gs[0, 1])
        ax_frame = fig3.add_subplot(gs[1, :])
        # plt.subplots_adjust(wspace=0, hspace=0)

        for ax in [ax_zoom, ax_frame]:
            ax.set_axis_off()

        time_here = camera_frame_to_trace_time(frame_id)
        ax_scope.set_title('Time: {0:.5f} s'.format(time_here))
        ax_scope.plot(scope_time, scope_droplet)
        ax_scope.axvline(x=time_here, color='C1')
        ax_scope.set_xlim(time_here - dt, time_here + dt)
        ax_scope.use_sticky_edges = False
        ax_scope.margins(0.05, tight=True)
        ax_scope.set_ylabel('Voltage on the droplet, V')
        ax_scope.set_xlabel('Time, s')
        ax_scope.set_ylim(-1, 9.5)
        ax2 = ax_scope.twinx()
        ax2.set_yticklabels(np.linspace(0, 1, 5))
        ax2.set_yticklabels(['   ']*5)
        ax2.set_ylabel('    ')


        img_for_show = np.copy(image)
        ax_frame.imshow(img_for_show, cmap='gray', vmin=18, vmax=180)
        # unsharp_mask(img_zoom(img_for_show - bkg_median), radius=1, amount=1, preserve_range=True),
        for_zoom = img_zoom(img_for_show - bkg_median)
        fzmax = np.max(np.abs(for_zoom))
        # for_zoom = filters.median(for_zoom, np.ones((3, 3)))
        for_zoom = unsharp_mask(for_zoom, radius=1.5, amount=0.5, preserve_range=True)
        ax_zoom.imshow(for_zoom,
                       cmap='gray', vmin=-60, vmax=60, interpolation='spline16') #
        # axarr[0].scatter(center[1], center[0], color='red', alpha=0.5)
        # axarr[2].imshow(diffimg2, cmap='seismic', vmin=-120, vmax=120)
        # prev_image = np.copy(image)
        # plt.show()
        fig3.savefig(frames_folder + '{0:08d}.png'.format(frame_id))
        plt.close('all')

        if frame_id % 200 == 0:
            bkg_median = get_median_bkg(frame_id)

    print('Speed was {0:.2f} frames/min'.format((frame_id_max - start_frame)/(time.time()-t0)*60))
    plt.show()

if __name__ == '__main__':
    # process_one_folder('E:/2020_nov_30/00_FASTS4_cleardrop_reg_160x60pics_good2_000000/',
    #                    scope_filename='scope_1-30clear132.csv',
    #                    start_frame=920, fps=30004.617,
    #                    center=(19, 126), size=8, time_scale=1.03719)

    process_one_folder('E:/FAST_CAM/2021_May_10_SDN_printing/FASTS4_2021-05-10_SDN_good_2ms_width_000000/',
                       scope_filename='New folder/5-10SDN-1ST3.csv',
                       start_frame=920, fps=5000,
                       center=(19, 126), size=8, time_scale=1.03719)

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