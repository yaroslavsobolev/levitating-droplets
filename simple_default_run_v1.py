from detect_tears import *
from analyze_detected_events import *


target_folder = (input('Enter path to target folder:') + '/').replace('\\', '/')
vpp = float(input('Input vpp (before amplifier):'))
freq = float(input('Input frequency (Hz):'))
process_one_folder(target_folder=target_folder,
                   raw_frames_suffix='',
                   thresh_1=25,
                   start_frame=1,
                   radius=False
                   )
analyze_detected_events(target_folder=target_folder,
                        raw_frames_suffix='',
                        gap_thresh=9,
                        amp_thresh=35,
                        amp_thresh_right=55,
                        amp_thresh_left=0,
                        do_space_filtering=False,
                        vp=vpp / 2,
                        freq=freq
                        )