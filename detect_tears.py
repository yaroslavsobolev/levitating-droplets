import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage import io
from scipy import ndimage as ndi
from skimage.filters import gaussian
import pickle
from skimage.feature import register_translation
from skimage.feature import canny
import os
import time
from glob import glob

# target_folder = 'F:/yankai_levitation_transfer_highspeed_videos/10Hz_300V_sinewave_1/whole_video_1/'
# raw_frames_suffix = 'FASTS4_000000/'

def process_one_folder(target_folder,
                        raw_frames_suffix = '',
                        thresh_1 = 15,
                        start_frame = 1,
                        do_canny=False,
                        do_figures = False,
                        low_threshold=30,
                        high_threshold=150,
                        radius=False,
                        center=False,
                        sum_0 = 31476558.0):
    video_folder = target_folder + raw_frames_suffix
    frames_folder = target_folder + 'frames_1/'
    events_folder = target_folder + 'events_pickle/'
    misc_folder = target_folder + 'misc/'
    for my_folder in [frames_folder, events_folder, misc_folder]:
        if not os.path.exists(my_folder):
            os.makedirs(my_folder)
    filelist = []
    # find all "imgNNN" subfolders
    img_subfolders = sorted(glob(video_folder + "img*/"))
    for video_subfolder in img_subfolders:
        for file in sorted(os.listdir(video_subfolder)):
            if file.endswith(".jpg"):
                filelist.append(os.path.join(video_subfolder, file))
    # filelist = sorted(filelist)
    print(filelist)
    print('Found {0} jpg frames.'.format(len(filelist)))
    frame_id_max = len(filelist) - 1

    def load_image(frame_id):
        # return np.sum(io.imread('{0}{1:07d}.jpg'.format(video_folder, frame_id)).astype(float), axis=2)
        return np.sum(io.imread(filelist[frame_id]).astype(float), axis=2)
    # sum_0 = np.sum(load_image(0))

    def load_normed(frame_id):
        image = load_image(frame_id)
        return image*sum_0/np.sum(image)

    # plt.imshow(load_normed(1), cmap='plasma', vmax=256*3)
    # plt.show()

    # Load picture and detect edges
    image = load_normed(1)
    edges = canny(image, sigma=3, low_threshold=low_threshold, high_threshold=high_threshold)

    fill_coins = ndi.binary_fill_holes(edges)
    if not center:
        center = ndi.center_of_mass(fill_coins)
    if not radius:
        radius = np.sqrt((np.sum(fill_coins)/np.pi))
    print(radius)
    np.save(misc_folder + 'radius', radius)
    np.save(misc_folder + 'center', center)
    np.save(misc_folder + 'first_frame', image)

    plt.scatter(center[1], center[0], color='red')
    image[edges]=1000
    plt.imshow(fill_coins)
    plt.show()
    int_rad = int(round(radius*1.1))
    ref_patch = image[int(round(center[0]))-int_rad:int(round(center[0]))+int_rad,
                      int(round(center[1]))-int_rad:int(round(center[1]))+int_rad]
    plt.imshow(ref_patch)
    plt.show()

    prev_image = load_normed(0)
    # center=(200, 200)
    t0 = time.time()
    for frame_id in range(start_frame,frame_id_max):
        print(frame_id)
        prev_image = load_normed(frame_id-1)
        image = load_normed(frame_id)

        target_patch = image[int(round(center[0])) - int_rad:int(round(center[0])) + int_rad,
                    int(round(center[1])) - int_rad:int(round(center[1])) + int_rad]
        shift, error, diffphase = register_translation(src_image=ref_patch, target_image=target_patch,
                                                       upsample_factor=4)
        if do_canny:
            edges = canny(image, sigma=3, low_threshold=20, high_threshold=110)
            fill_coins = ndi.binary_fill_holes(edges)#, origin=[int(round(center[0])), int(round(center[1]))])
            center_canny = ndi.center_of_mass(fill_coins)
        center -= np.array(shift)
        diffimg = image-prev_image
        diffimg2 = gaussian(diffimg, sigma=(2,2))

        if do_figures:
            fig, axarr = plt.subplots(ncols=4, nrows=1, figsize=(14, 4))
            plt.subplots_adjust(wspace=0, hspace=0)
            for ax in axarr:
                ax.set_axis_off()
            img_for_show = np.copy(image)
            axarr[0].imshow(img_for_show, cmap='gray')
            axarr[0].scatter(center[1], center[0], color='red', alpha=0.5)

            if do_canny:
                axarr[1].imshow(fill_coins)
                axarr[1].scatter(center[1], center[0], color='blue', alpha=0.5)
                axarr[1].scatter(center_canny[1], center_canny[0], s=40, marker='x', color='black', alpha=0.5)
            axarr[2].imshow(diffimg2, cmap='seismic', vmin=-120, vmax=120)
        diffimg3 = gaussian(np.abs(diffimg), sigma=(2,2))
        diffimg3[:int(round(center[0])), :] = 0
        if do_figures:
            im = axarr[3].imshow(diffimg3, vmin=0, vmax=150)

        loc_max = np.unravel_index(np.argmax(diffimg3, axis=None), diffimg3.shape)

        if np.max(diffimg3) > thresh_1:
            for_pickle = [frame_id, np.max(diffimg3), loc_max, center]
            pickle.dump(for_pickle, open(target_folder + 'events_pickle/{0:07d}.p'.format(frame_id), "wb"))
            if do_figures:
                for ax in axarr:
                    e1 = patches.Ellipse((loc_max[1], loc_max[0]), 50, 50,
                                         angle=0, linewidth=2, fill=False, zorder=2, color='red')
                    ax.add_patch(e1)
        if do_figures:
            axarr[3].scatter(center[1], center[0], color='red')
            plt.colorbar(im)
            fig.savefig('{0}frames_1/{1:07d}'.format(target_folder, frame_id))
            # plt.show()
            plt.close(fig)
        prev_image = np.copy(image)
    print('Speed was {0:.2f} frames/min'.format((frame_id_max - start_frame)/(time.time()-t0)*60))
    plt.show()

if __name__ == '__main__':
    # process_one_folder('H:/FASTS4_2020-10-15_1409_000000/img0000/',
    #                    raw_frames_suffix='',
    #                    thresh_1=25,
    #                    start_frame=1,
    #                    radius=50,
    #                    center=(90,150),
    #                    do_figures=False
    #                    )
    process_one_folder('E:/fastcam/2020_nov_10/FASTS4_11-10_1000rpm_000000/',
                       raw_frames_suffix='',
                       thresh_1=20,
                       start_frame=1,
                       radius=35,
                       center=(50,110),
                       do_figures=False,
                       sum_0 = 31476558.0/8
                       )