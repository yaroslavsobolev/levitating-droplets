from detect_tears import *
from analyze_detected_events import *

# process_one_folder('F:/yankai_levitation_transfer_highspeed_videos/'
#                         '2020_sep_20/1 droplet2 voltage changing_all at 10hz/'
#                         'FASTS4_000002_2300fps_2.5Vp_10hz_yesprinting/',
#                    raw_frames_suffix = '',
#                    thresh_1 = 15,
#                    start_frame = 4150)
# analyze_detected_events('F:/yankai_levitation_transfer_highspeed_videos/'
#                         '2020_sep_20/1 droplet2 voltage changing_all at 10hz/'
#                         'FASTS4_000002_2300fps_2.5Vp_10hz_yesprinting/',
#                         raw_frames_suffix = '',
#                         gap_thresh = 25,
#                         amp_thresh = 15,
#                         x_locs = [-65.45, 70.2],
#                         x_loc_width = 20,
#                         do_space_filtering = True)

# process_one_folder('F:/yankai_levitation_transfer_highspeed_videos/'
#                         '2020_sep_20/1 droplet2 voltage changing_all at 10hz/'
#                         'FASTS4_000003_2300fps2.25Vp_10hz_yesprinting/',
#                    raw_frames_suffix = '',
#                    thresh_1 = 15,
#                    start_frame = 9360,
#                    do_canny=False,
#                    do_figures=False
#                    )
# analyze_detected_events('F:/yankai_levitation_transfer_highspeed_videos/'
#                         '2020_sep_20/1 droplet2 voltage changing_all at 10hz/'
#                         'FASTS4_000003_2300fps2.25Vp_10hz_yesprinting/',
#                         raw_frames_suffix = '',
#                         gap_thresh = 9,
#                         amp_thresh = 15,
#                         do_space_filtering = True,
#                         x_locs=[-63.9, 69.38],
#                         x_loc_width=10,
#                         y_locs = [94, 86],
#                         y_loc_width = [15, 15]
#                         )

# process_one_folder('F:/yankai_levitation_transfer_highspeed_videos/'
#                         '2020_sep_23/Fluorimetry/'
#                         'FASTS4_2vp-1_000000/',
#                    raw_frames_suffix = '',
#                    thresh_1 = 15,
#                    start_frame = 1,
#                    do_canny=False,
#                    do_figures=False,
#                    radius=325-181
#                    )
# analyze_detected_events('F:/yankai_levitation_transfer_highspeed_videos/'
#                         '2020_sep_23/Fluorimetry/'
#                         'FASTS4_2vp-1_000000/',
#                         raw_frames_suffix = '',
#                         gap_thresh = 9,
#                         amp_thresh = 35,
#                         amp_thresh_right=55,
#                         amp_thresh_left=0,
#                         do_space_filtering = True,
#                         x_locs=[-76.0, 68.0],
#                         x_loc_width=10,
#                         y_locs = [112, 121],
#                         y_loc_width = [15, 15],
#                         shift_thresh = 20,
#                         n_event_thresh=0
#                         )

# process_one_folder('F:/yankai_levitation_transfer_highspeed_videos/'
#                         '2020_sep_23/Fluorimetry/'
#                         'FASTS4_2vp-2_000000/',
#                    raw_frames_suffix = '',
#                    thresh_1 = 25,
#                    start_frame = 1,
#                    do_canny=False,
#                    do_figures=False,
#                    radius=325-181
#                    )

# process_one_folder('F:/yankai_levitation_transfer_highspeed_videos/'
#                         '2020_sep_23/Fluorimetry/'
#                         'FASTS4_1.75vp-1_000000/',
#                    raw_frames_suffix = '',
#                    thresh_1 = 25,
#                    start_frame = 15417,
#                    do_canny=False,
#                    do_figures=False,
#                    radius=False
#                    )
# analyze_detected_events('F:/yankai_levitation_transfer_highspeed_videos/'
#                         '2020_sep_23/Fluorimetry/'
#                         'FASTS4_1.75vp-1_000000/',
#                         raw_frames_suffix = '',
#                         gap_thresh = 9,
#                         amp_thresh = 35,
#                         amp_thresh_right=55,
#                         amp_thresh_left=0,
#                         do_space_filtering = False,
#                         x_locs=[-76.0, 68.0],
#                         x_loc_width=10,
#                         y_locs = [112, 121],
#                         y_loc_width = [15, 15],
#                         n_event_thresh=0
#                         )

# process_one_folder('F:/yankai_levitation_transfer_highspeed_videos/'
#                         '2020_sep_25/'
#                         'FASTS4_1.25Vp10hz_Synced_000000/',
#                    raw_frames_suffix = '',
#                    thresh_1 = 25,
#                    start_frame = 9300,
#                    do_canny=False,
#                    do_figures=False,
#                    radius=False
#                    )
# analyze_detected_events('F:/yankai_levitation_transfer_highspeed_videos/'
#                         '2020_sep_25/'
#                         'FASTS4_1.25Vp10hz_Synced_000000/',
#                         raw_frames_suffix = '',
#                         gap_thresh = 9,
#                         amp_thresh = 35,
#                         amp_thresh_right=55,
#                         amp_thresh_left=0,
#                         do_space_filtering = False,
#                         x_locs=[-76.0, 68.0],
#                         x_loc_width=10,
#                         y_locs = [112, 121],
#                         y_loc_width = [15, 15],
#                         n_event_thresh=0,
#                         vp = 2.5/2,
#                         fps=2300,
#                         limc=2.5/2*1.1*100
#                         )

# process_one_folder('F:/yankai_levitation_transfer_highspeed_videos/'
#                    '2020_sep_25/'
#                    'FASTS4_2300fps_1.3Vpp10hz_Synced-4_000001/',
#                    raw_frames_suffix='',
#                    thresh_1=25,
#                    start_frame=7411,
#                    radius=False
#                    )
# analyze_detected_events('F:/yankai_levitation_transfer_highspeed_videos/'
#                         '2020_sep_25/'
#                         'FASTS4_2300fps_1.3Vpp10hz_Synced-4_000001/',
#                         raw_frames_suffix='',
#                         gap_thresh=9,
#                         amp_thresh=35,
#                         amp_thresh_right=55,
#                         amp_thresh_left=0,
#                         do_space_filtering=False,
#                         vp=1.3 / 2,
#                         freq=10,
#                         limc=2.5 / 2 * 1.1 * 100
#                         )

# process_one_folder('H:/FASTS4_2020-10-15_1409_000000/img0000/',
#                    raw_frames_suffix='',
#                    thresh_1=25,
#                    start_frame=1,
#                    radius=50,
#                    center=(90, 150),
#                    do_figures=False
#                    )

# process_one_folder('E:/fastcam/2020_nov_10/FASTS4_11-10_850rpm_000000/',
#                    raw_frames_suffix='',
#                    thresh_1=20,
#                    start_frame=1,
#                    radius=35,
#                    center=(50, 110),
#                    do_figures=False,
#                    sum_0=31476558.0 / 8
#                    )
# analyze_detected_events('E:/fastcam/2020_nov_10/FASTS4_11-10_850rpm_000000/',
#                         raw_frames_suffix='',
#                         gap_thresh=9,
#                         amp_thresh=20,
#                         amp_thresh_right=55,
#                         amp_thresh_left=0,
#                         do_space_filtering=True,
#                         x_locs=[-78.5, 4],
#                         x_loc_width=5,
#                         y_locs = [20, 45],
#                         y_loc_width = [3, 3],
#                         vp=2.25 / 2,
#                         freq=150,
#                         limc=2.5 / 2 * 1.1 * 100,
#                         last_event=3000
#                         )

# process_one_folder('E:/fastcam/2020_nov_10/FASTS4_11-10_900rpm_000000/',
#                    raw_frames_suffix='',
#                    thresh_1=20,
#                    start_frame=1,
#                    radius=35,
#                    center=(50, 110),
#                    do_figures=False,
#                    sum_0=31476558.0 / 8
#                    )
# analyze_detected_events('E:/fastcam/2020_nov_10/FASTS4_11-10_900rpm_000000/',
#                         raw_frames_suffix='',
#                         gap_thresh=9,
#                         amp_thresh=20,
#                         amp_thresh_right=55,
#                         amp_thresh_left=0,
#                         do_space_filtering=True,
#                         x_locs=[-76.7, 8],
#                         x_loc_width=2,
#                         y_locs = [8.44, 29],
#                         y_loc_width = [3, 3],
#                         vp=2.25 / 2,
#                         freq=150,
#                         limc=2.5 / 2 * 1.1 * 100,
#                         last_event=6000
#                         )
# process_one_folder('E:/fastcam/2020_nov_10/FASTS4_11-10_950rpm_000001/',
#                    raw_frames_suffix='',
#                    thresh_1=20,
#                    start_frame=1,
#                    radius=35,
#                    center=(50, 110),
#                    do_figures=False,
#                    sum_0=31476558.0 / 8
#                    )
# analyze_detected_events('E:/fastcam/2020_nov_10/FASTS4_11-10_950rpm_000001/',
#                         raw_frames_suffix='',
#                         gap_thresh=9,
#                         amp_thresh=20,
#                         amp_thresh_right=55,
#                         amp_thresh_left=0,
#                         do_space_filtering=True,
#                         x_locs=[-74.5, 18],
#                         x_loc_width=4,
#                         y_locs = [8.40, 36],
#                         y_loc_width = [3, 3],
#                         vp=2.00 / 2,
#                         freq=150,
#                         limc=2.5 / 2 * 1.1 * 100,
#                         last_event=3000
#                         )
# process_one_folder('E:/fastcam/2020_nov_10/FASTS4_11-10_1000rpm_000000/',
#                    raw_frames_suffix='',
#                    thresh_1=20,
#                    start_frame=1,
#                    radius=35,
#                    center=(50, 110),
#                    do_figures=False,
#                    sum_0=31476558.0 / 8
#                    )
analyze_detected_events('E:/fastcam/2020_nov_10/FASTS4_11-10_1000rpm_000000/',
                        raw_frames_suffix='',
                        gap_thresh=9,
                        amp_thresh=20,
                        amp_thresh_right=55,
                        amp_thresh_left=0,
                        do_space_filtering=True,
                        x_locs=[-58, 33],
                        x_loc_width=3,
                        y_locs = [12.7, 36],
                        y_loc_width = [3, 3],
                        vp=2.00 / 2,
                        freq=150,
                        limc=2.5 / 2 * 1.1 * 100,
                        last_event=3000
                        )