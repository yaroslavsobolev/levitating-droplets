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
analyze_detected_events('F:/yankai_levitation_transfer_highspeed_videos/'
                       '2020_oct_15/'
                       'FASTS4_2020-10-15_1409_000000/img0000/',
                        raw_frames_suffix='',
                        gap_thresh=9,
                        amp_thresh=35,
                        amp_thresh_right=55,
                        amp_thresh_left=0,
                        do_space_filtering=True,
                        x_locs=[-75.0, 86.0],
                        x_loc_width=6,
                        y_locs = [14, 13],
                        y_loc_width = [8, 7],
                        vp=2 / 2,
                        freq=10,
                        limc=2.5 / 2 * 1.1 * 100
                        )