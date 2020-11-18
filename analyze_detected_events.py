import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import configparser

def analyze_detected_events(target_folder, raw_frames_suffix='', gap_thresh=10, amp_thresh=15,
                            do_space_filtering=False, x_locs=[-65.45, 70.2], x_loc_width=20,
                            y_locs=[94, 86], y_loc_width=[15, 15],
                            shift_thresh = 30,
                            n_event_thresh = 0,
                            amp_thresh_right = 0,
                            amp_thresh_left = 0,
                            vp = 2.5,
                            freq = 10,
                            fps=False,
                            limc=False,
                            last_event=False):
    video_folder = target_folder + raw_frames_suffix
    misc_folder = target_folder + 'misc/'
    figures_folder = target_folder + 'figures/'
    for my_folder in [figures_folder]:
        if not os.path.exists(my_folder):
            os.makedirs(my_folder)
    config = configparser.ConfigParser()
    config.read(video_folder + 'metadata.txt')
    if not fps:
        fps = int(config['Record']['fps'])
    print('FPS is: {0}'.format(fps))

    radius = np.load(misc_folder + 'radius.npy')
    center_in_first_frame = np.load(misc_folder + 'center.npy')
    first_frame = np.load(misc_folder + 'first_frame.npy')


    data = []
    all_pickle_files = os.listdir(target_folder + 'events_pickle/')
    if last_event:
        all_pickle_files = all_pickle_files[:last_event]
    for filename in all_pickle_files:
        with open(os.path.join(target_folder + 'events_pickle/', filename), 'rb') as f: # open in readonly mode
            for_pickle = pickle.load(f)
        data.append(for_pickle)
        print(for_pickle)

    # group the events into discharges
    discharge_on = False
    discharges = []
    current_discharge = []
    last_discharge_frame = 0
    event_poses = []
    for event_id in range(len(data)):
        amplitude = data[event_id][1]
        event_pos = np.array(data[event_id][2]) - np.array(data[event_id][3])
        event_poses.append(event_pos)
        x_pos_wrt_center = event_pos[1]
        y_pos_wrt_center = event_pos[0]
        if do_space_filtering:
            if (x_pos_wrt_center > x_locs[0] - x_loc_width and \
                x_pos_wrt_center < x_locs[0] + x_loc_width and \
                y_pos_wrt_center > y_locs[0] - y_loc_width[0] and \
                y_pos_wrt_center < y_locs[0] + y_loc_width[1]) or \
                (x_pos_wrt_center > x_locs[1] - x_loc_width and \
                 x_pos_wrt_center < x_locs[1] + x_loc_width and \
                 y_pos_wrt_center > y_locs[1] - y_loc_width[1] and \
                 y_pos_wrt_center < y_locs[1] + y_loc_width[1]):
                print('')
            else:
                print('Filtered out by coordinate.')
                continue
        if amplitude < amp_thresh:
            continue

        if x_pos_wrt_center > 0:
            if amplitude < amp_thresh_right:
                continue
        else:
            if amplitude < amp_thresh_left:
                continue

        if data[event_id][0] - last_discharge_frame < gap_thresh:#) and \
            # np.linalg.norm(x_pos_wrt_center - last_discharge_pos) < shift_thresh:
            current_discharge.append(data[event_id])
        else:
            if current_discharge:
                discharges.append(current_discharge)
            current_discharge = []
            current_discharge.append(data[event_id])

        last_discharge_frame = data[event_id][0]
        last_discharge_pos = np.copy(x_pos_wrt_center)
        # if event_id > 200:
        #     break

    # fig0 = plt.figure(4)
    event_poses = np.array(event_poses)
    # plt.scatter(event_poses[:,1], event_poses[:,0], s=3, alpha=0.5)
    # plt.show()

    discharge_stats = []
    for discharge in discharges:
        amp_max = max([event[1] for event in discharge])
        # filter out the false positives
        if amp_max < amp_thresh:
            continue
        start_frame = min([event[0] for event in discharge])
        end_frame = max([event[0] for event in discharge])
        duration = end_frame - start_frame + 1
        n_events = len(discharge)
        if n_events <= n_event_thresh:
            continue
        event_positions = np.array([np.array(event[2]) - np.array(event[3]) for event in discharge])
        mean_pos = np.mean(event_positions, axis=0)
        discharge_stats.append([start_frame, amp_max, n_events, duration, mean_pos[0], mean_pos[1]])

    discharge_stats = np.array(discharge_stats)
    dist_to_prev = np.diff(discharge_stats[:,0])
    fig, axarr = plt.subplots(5, 1, sharex=True, figsize=(12,8))
    factor = fps
    # factor = 1
    colors = ['red' if xpos > 0 else 'blue' for xpos in discharge_stats[:, 5]]
    axarr[0].plot(discharge_stats[:,0]/factor, discharge_stats[:,1])
    axarr[1].plot(discharge_stats[:,0]/factor, discharge_stats[:,2])
    ts = np.linspace(0, np.max(discharge_stats[:, 0]) / factor, 10000)
    axarr[2].plot(ts, 100*vp*np.sin(2*np.pi*ts*freq), color='black', alpha=0.5)
    for i,x in enumerate(discharge_stats[:, 0]):
        # axarr[2].axvline(x=x/factor, color=colors[i])
        axarr[2].scatter(x / factor, 100*vp*np.sin(2*np.pi*x/factor*freq), s=10,
                      color=colors[i])
    axarr[2].axhline(y=0, color='grey')
    axarr[3].plot(discharge_stats[1:,0]/factor, dist_to_prev/fps*1000)
    axarr[4].scatter(discharge_stats[:,0]/factor, discharge_stats[:,5], s=100, marker="|", color=colors)

    axarr[0].set_ylabel('Prominence, a.u.')
    axarr[1].set_ylabel('Number of frames')
    axarr[2].set_ylabel('Applied potential, V')
    axarr[3].set_ylabel('Time delay since\n''previous, ms')
    axarr[4].set_ylabel('Position w.r.t.\ndroplet center, px')
    axarr[4].set_xlabel('Time, s')
    plt.tight_layout()

    f2 = plt.figure(2)
    plt.hist(discharge_stats[:,5], bins=100)
    plt.xlabel('X coordinate of the event w.r.t droplet center, pixels')
    plt.ylabel('Number of events')
    plt.title('Histogram of the horizontal event positions w.r.t. droplet center')
    # plt.show()
    print('Found {0} events.'.format(discharge_stats.shape[0]))
    np.savetxt(misc_folder + 'n_events.txt', np.array([discharge_stats.shape[0]]))

    f3 = plt.figure(3)
    plt.imshow(first_frame, cmap='gray')
    plt.scatter(discharge_stats[:,5] + center_in_first_frame[1], discharge_stats[:,4]+center_in_first_frame[0],
                s=3, alpha=0.2, color=colors)
    plt.axis('equal')

    f4 = plt.figure(4)
    # plt.imshow(first_frame, cmap='gray')
    plt.scatter(discharge_stats[:,5], discharge_stats[:,4],
                s=3, alpha=0.5, color=colors)
    plt.axis('equal')
    plt.xlabel('X coordinate of the event w.r.t droplet center, pixels')
    plt.xlabel('Y coordinate of the event w.r.t droplet center, pixels')


    # plotting the scatterplot and hists
    # definitions for the axes
    left, width = 0.11, 0.75
    bottom, height = 0.1, 0.75
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.1]
    rect_histy = [left + width + spacing, bottom, 0.1, height]

    # start with a rectangular Figure
    f5 = plt.figure(5, figsize=(6, 6))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the scatter plot:
    ts = np.linspace(0, 1/freq, 1000)
    ax_scatter.plot(ts, 100*vp*np.sin(2*np.pi*ts*freq), color='black')
    crossings = []
    xs = []
    for i,x in enumerate(discharge_stats[:, 0]):
        time_within_period = (x / factor) % (1/freq)
        # axarr[2].axvline(x=x/factor, color=colors[i])
        crossing = 100*vp*np.sin(2*np.pi*x/factor*freq)
        crossings.append(crossing)
        xs.append(time_within_period)
        ax_scatter.scatter(time_within_period, crossing, s=30,
                      color=colors[i], alpha=0.2)
    xs = np.array(xs)
    crossings = np.array(crossings)
    np.savetxt(misc_folder + 'crossings.txt', crossings)
    print('Mean threshhold voltage at events: {0:.2f} V'.format(np.mean(np.abs(crossings))))
    # f5.set_title('Locations of events at the respective field cycle')
    ax_scatter.set_ylabel('Applied potential, V')
    ax_scatter.set_xlabel('Time, s')
    ax_scatter.axhline(y=0, color='grey')
    # ax_scatter.scatter(xs, crossings)

    # now determine nice limits by hand:
    binwidthx = 1/freq/200
    limx_max = 1/freq
    ax_scatter.set_xlim((0, limx_max))
    binsx = np.arange(0, limx_max + binwidthx, binwidthx)

    binwidthc = (np.std(crossings[crossings>0]) + np.std(crossings[crossings<0]))
    # limc = np.ceil(np.abs(crossings).max() / binwidthc) * binwidthc
    if not limc:
        limc = 1.1*100*vp
    ax_scatter.set_ylim((-limc, limc))
    binsc = np.arange(-limc, limc + binwidthc, binwidthc)

    ax_histx.hist(xs, bins=binsx, color='grey')
    ax_histy.hist(crossings, bins=binsc, orientation='horizontal', color='grey')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    # ax_histy.xaxis.set_label_position("top")
    ax_histy.xaxis.tick_top()
    # ax_histy.set_xticks(ax_histy.get_xticks()[1:])

    # plt.show()

    f6, axarr2 = plt.subplots(2,1)
    axarr2[0].hist(crossings[crossings > 0])
    axarr2[1].hist(crossings[crossings <= 0])
    for ax in axarr2:
        ax.set_xlabel('Potential at discharge, V')
        ax.set_ylabel('Number of events')
    plt.tight_layout()

    def get_binomial_runs(N=100000):
        return np.random.randint(2, size=N)

    def count_runs(xs):
        seqlens = []
        count = 0
        for x in xs:
            if x:
                count += 1
            else:
                if count > 0:
                    seqlens.append(count)
                count = 0
        return seqlens

    f7 = plt.figure(7, figsize=(6,6))
    plt.style.use('seaborn-deep')
    plt.hist([count_runs(get_binomial_runs()), count_runs(discharge_stats[:, 5] > 0)],
             bins=-0.5+np.arange(8), density=True, alpha=0.5, label=['Binomial','Experiment'])
    # plt.hist(,
    #          bins=-0.5+np.arange(8), density=True, alpha=0.5, label='Experiment')
    # plt.xticks(np.arange(8))
    plt.ylabel('Probability density')
    plt.xlabel('Number of subsequent events on the same side')
    plt.legend()

    for i, f in enumerate([fig, f2, f3, f4, f5, f6, f7]):
        f.savefig(figures_folder + 'figure_{0}.png'.format(i+1), dpi=300)
    plt.show()

if __name__ == '__main__':
    print('This is __main__')