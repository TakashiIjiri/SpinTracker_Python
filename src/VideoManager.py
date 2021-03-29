
from threading import Lock
from tkinter import filedialog
import cv2
import time

import tmesh
import tutil
import numpy as np



"""
class VideoManager
# Singleton class
# manage video and tracking data
"""
class VideoManager:

    #class variables for singleton
    __instance = None
    __lock = Lock()

    def __new__(cls):
        # None can init this class
        raise NotImplementedError('Cannot initialize via Constructor')

    @classmethod
    def __internal_new__(cls):
        return super().__new__(cls)

    @classmethod
    def get_inst( cls ):
        if cls.__instance == None:
            with cls.__lock:
                if cls.__instance == None:
                    cls.__instance = cls.__internal_new__()
                    cls.__instance.__init__()
        return cls.__instance



    # constructor (called only from get_inst())
    def __init__(self) :

        print("------------1. Load Video------------")
        if False :
            ftype = [('load mp4 file','*.mp4')]
            self.__video_path = filedialog.askopenfilename(filetypes = ftype , initialdir = "./")
            if self.__video_path == None :
                exit()
        #else : self.__video_path = "../video/video3.mp4" # デバッグ用
        else : self.__video_path = "../video/video.mp4" # デバッグ用

        self.__frames = tutil.load_video(self.__video_path, 600) # np.float32
        n, h, w = self.__frames.shape
        self.__roi_frames      = np.zeros((n,int(h/4),int(w/4)), dtype=np.float32 )
        self.__roi_frames_bin  = np.zeros((n,int(h/4),int(w/4)), dtype=np.uint8   )

        # load sphere mesh -> normalize the vertext position (spin dir candidate)
        self.__sphere_model = tmesh.TMesh( init_as="Obj", fname="./sphere3.obj")
        norms = np.linalg.norm(self.__sphere_model.verts, axis=1).reshape(-1,1)
        self.__sphere_model.verts  /= norms
        self.__sphere_model.set_projection_texture()

        # parameters used for tracking 
        self.__current_frame_idx = 0
        self.__roi_rect = np.array( [1/4*w, 1/4*h, 3/4*w, 3/4*h])
        self.__ZOOM_WIN_R = 50
        self.__zoom_win_c = [w/2, h/2]
        self.__ball_r1 = 20
        self.__ball_r2 = 15

        # tracking info and ball clips computed by "run_tracking()"
        self.__tempmatch_cxcy = np.zeros((n,2), dtype=np.int32  )
        self.__hough_xyr      = np.zeros((n,3), dtype=np.float32)
        self.__ball_clips      = np.zeros((n,int(50),int(50)), dtype=np.float32 )
        self.__ball_clips_diff = np.zeros((n,int(50),int(50)), dtype=np.float32 )
        self.__ball_clip_ave   = np.zeros((  int(50),int(50)), dtype=np.float32 )
        self.__ball_clip_mask  = np.zeros((  int(50),int(50)), dtype=np.float32 )

        # RPS (Revolution per seconds) candidates by "run_estimate_spinspeed()"
        self.__estimated_periods = np.zeros(0)
        self.__estimated_RPSs = np.zeros(0)

        # spin axis computed by "run_estimate_spinaxis"
        self.__spin_axis = np.array([.0,.1,.0], dtype=np.float32)
        self.__costs_for_allaxis = np.zeros(self.__sphere_model.verts.shape[0])
        self.__spin_period = 1000

    def get_frame_uint8(self, idx) :
        if idx < 0 or self.__frames.shape[0] <= idx:
            return np.uint8( np.zeros_like(self.__frames[0]) )
        return np.uint8( self.__frames[idx] )

    def get_frame_size(self):
        if self.__frames.shape[0] == 0 :
            return 0, 0
        return self.__frames.shape[1], self.__frames.shape[2] # return height, width of frame

    def get_roi_frame_bin_uint8(self, idx) :
        if idx < 0 or self.__roi_frames_bin.shape[0] <= idx:
            return np.zeros_like( self.__roi_frames_bin[0], dtype=np.uint8 )
        return np.uint8( self.__roi_frames_bin[idx] )

    def get_ball_clip_uint8(self, idx) :
        if idx < 0 or self.__ball_clips.shape[0] <= idx:
            return np.zeros_like( self.__ball_clips[0], dtype=np.uint8 )
        return np.uint8( self.__ball_clips[idx] )

    def get_sphere_model(self) :
        return self.__sphere_model

    def get_spin_axis(self):
        return self.__spin_axis

    def get_ball_clip_diff_uint8(self, idx) :
        if idx < 0 or self.__ball_clips_diff.shape[0] <= idx:
            return np.zeros_like( self.__ball_clips_diff[0], dtype=np.uint8 )
        bc_1 =  self.__ball_clips_diff[idx].copy()
        bc_2 = -self.__ball_clips_diff[idx].copy()
        bc_1[bc_1<0] = 0
        bc_2[bc_2<0] = 0
        h,w = bc_2.shape
        img = np.zeros((h,w,3), dtype=np.uint8)
        img[:,:,0] = np.uint8(3 * bc_2)
        img[:,:,1] = np.uint8(3 * bc_1)
        img[:,:,2] = np.uint8(3 * bc_1)
        return img

    def get_ball_clip_average_uint8(self) :
        return np.uint8( self.__ball_clip_ave )

    def get_ball_clip_mask_uint8(self):
        img = self.__ball_clip_mask.copy()
        max_v = np.max(img)
        if max_v > 0 : img = img / max_v * 255
        return np.uint8( img )

    def get_axis_estim_costs(self) : 
        return self.__costs_for_allaxis

    def num_frames(self) :
        return len(self.__frames)

    def set_current_frame_idx(self, idx):
        self.__current_frame_idx = idx

    def increment_current_frame_idx(self):
        self.__current_frame_idx += 1
        n, h, w = self.__frames.shape
        if self.__current_frame_idx >= n :
            self.__current_frame_idx = n - 1

    def decrement_current_frame_idx(self):
        self.__current_frame_idx -= 1
        if self.__current_frame_idx < 0:
            self.__current_frame_idx = 0


    def get_current_frame_idx(self):
        return self.__current_frame_idx

    def get_roi_rect(self):
        return self.__roi_rect

    # rect is an array of (x0, y0, x1, y1)
    def set_roi_rect(self, rect) :
        n, h, w = self.__frames.shape
        self.__roi_rect[0] = max(0  , min(rect[0], rect[2]))
        self.__roi_rect[1] = max(0  , min(rect[1], rect[3]))
        self.__roi_rect[2] = min(w-1, max(rect[0], rect[2]))
        self.__roi_rect[3] = min(h-1, max(rect[1], rect[3]))

    def set_ball_radius(self, r1, r2):
        self.__ball_r1 = r1
        self.__ball_r2 = r2

    def get_ball_radius(self):
        return self.__ball_r1, self.__ball_r2

    def set_zoom_win_center(self, c):
        x, y = int(c[0]), int(c[1])
        n, h, w = self.__frames.shape
        R = self.__ZOOM_WIN_R
        if x - R < 0 : x = R
        if y - R < 0 : y = R
        if x + R >= w : x = w - 1 - R
        if y + R >= h : y = h - 1 - R
        self.__zoom_win_c = np.array([x,y])

    def get_zoom_win_rect(self):
        x0 = self.__zoom_win_c[0] - self.__ZOOM_WIN_R
        x1 = self.__zoom_win_c[0] + self.__ZOOM_WIN_R
        y0 = self.__zoom_win_c[1] - self.__ZOOM_WIN_R
        y1 = self.__zoom_win_c[1] + self.__ZOOM_WIN_R
        return np.array([x0, y0, x1, y1])

    # generate multiple circlular templates
    # templates[ 0 ] has radius self.ball_r1
    # templates[n-1] has radius self.ball_r1
    # templates[i]   interplates r1~r2
    def __gen_ball_templates(self, num_templates) :
        r = max(self.__ball_r1, self.__ball_r2)
        temp_radi = int( r * 3 / 2 )
        temp_size = 2 * temp_radi + 1
        dR  = (self.__ball_r2 - self.__ball_r1) / ( num_templates - 1)

        ball_templates = np.zeros( (num_templates, temp_size, temp_size), dtype=np.uint8 )
        for i in range(num_templates) :
            radi = int(self.__ball_r1 + dR * i)
            cv2.circle(ball_templates[i], (temp_radi, temp_radi), radi, (255, 255, 255), thickness=-1)
        return ball_templates

    def get_tempmatch_rect(self, idx):
        if idx < 0 or self.__roi_frames_bin.shape[0] <= idx:
            return np.zeros(4, dtype = np.int32)
        temp_radi = int( self.__ball_r1 * 3 / 2 )
        x0 = self.__tempmatch_cxcy[idx, 0] - temp_radi
        x1 = self.__tempmatch_cxcy[idx, 0] + temp_radi + 1
        y0 = self.__tempmatch_cxcy[idx, 1] - temp_radi
        y1 = self.__tempmatch_cxcy[idx, 1] + temp_radi + 1
        return np.array([x0, y0, x1, y1])

    def get_hough_cxcyr (self, idx):
        if idx < 0 or self.__roi_frames_bin.shape[0] <= idx:
            return np.zeros(3, dtype = np.float32)
        return self.__hough_xyr[idx]

    def update_ballclip_mask( self, r1, r2, mask_angle, mask_rate) :
        r = max(r1,r2)
        self.__ball_clip_mask = tutil.gen_mask_image(r, mask_angle, mask_rate)


    
    def run_taracking_and_spinestimation(
        self, 
        radius_release, radius_catch     , 
        bkgrnd_mode   , bkgrnd_thresh,
        morpho_size   , tempmatch_step   ,
        mask_angle    , mask_rate        ,
        video_fps
    ):
        time_0 = time.perf_counter()

        self.__ball_r1 = radius_release
        self.__ball_r2 = radius_catch
        rect = np.int32(self.__roi_rect)
        self.__roi_frames = self.__frames[:,rect[1]:rect[3], rect[0]:rect[2]]

        # 1. tracking by template matching
        center_seq, start_i, end_i, frames_bin = tutil.t_trackball_temp_match(
                                    self.__roi_frames, 
                                    self.__ball_r1   , self.__ball_r2   ,
                                    bkgrnd_mode, bkgrnd_thresh, tempmatch_step )
        self.__tempmatch_cxcy = center_seq
        self.__roi_frames_bin = frames_bin
        # trim first 10% frames
        TRIM_RATE = 0.1
        start_i  += int( (end_i - start_i) * TRIM_RATE)
        time_1 = time.perf_counter()

        #2. tracking by hough transform 
        self.__hough_xyr = tutil.t_trackball_hough( 
                                    self.__roi_frames_bin, 
                                    self.__ball_r1, self.__ball_r2,
                                    self.__tempmatch_cxcy, 
                                    start_i, end_i) 
        time_2 = time.perf_counter()

        if end_i - start_i < 32 :
            print("the system fails to track the ball well")
            return

        #3. generate ballclip
        self.__ball_clips = tutil.t_generate_ballclip2( 
                                    self.__roi_frames, 
                                    self.__ball_r1, 
                                    self.__ball_r2, 
                                    self.__hough_xyr,
                                    start_i, end_i)
        time_3 = time.perf_counter()

        # 4. generate mask and spin speed estimation
        w = self.__ball_clips.shape[1]
        self.__ball_clip_mask = tutil.gen_mask_image( w//2, mask_angle, mask_rate)

        max_rps = 56 #3360 RPM
        estim_periods, estim_RPSs = tutil.t_estimate_spinspeed ( 
                                    self.__ball_clips[ start_i : end_i + 1 ], 
                                    self.__ball_clip_mask,
                                    video_fps, 
                                    max_rps)
        self.__estimated_RPSs = estim_RPSs
        self.__estimated_periods = estim_periods
        time_4 = time.perf_counter()

        print("found spin period candidates", self.__estimated_periods)
        if self.__estimated_periods.shape[0] <= 0 :
            print("system failes to estimte rotation speed")
            return

        #5. prepare ballclip_diff (ballclip - ballclip_ave)
        trgt_frms = max( 16, int( np.min(self.__estimated_periods) ) ) * 2
        trgt_frms = min(trgt_frms, self.__ball_clips.shape[0] )

        bc_ave, bc_diff = tutil.t_gen_ballclip_diff( 
                                    self.__ball_clips[start_i : start_i + trgt_frms, :,:], 
                                    BLUR_RATE = 0.1)
        
        self.__ball_clip_ave = bc_ave
        self.__ball_clips_diff = np.zeros_like(self.__ball_clips, dtype=np.float64)
        self.__ball_clips_diff[ start_i : start_i + trgt_frms ] = bc_diff
        time_5 = time.perf_counter()

        # 6. spin axis estimation
        optim_axis, optim_period, costs_for_allaxes = tutil.t_estimate_spinaxis(
                                    self.__ball_clips_diff[ start_i : start_i + trgt_frms ],
                                    self.__ball_clip_mask,
                                    self.__estimated_periods,
                                    self.__sphere_model.verts )
        self.__spin_axis = optim_axis
        self.__spin_period = optim_period
        self.__costs_for_allaxis = tutil.t_normalize_1d(costs_for_allaxes)
        time_6 = time.perf_counter()

        print("estimation time: step1~step6[sec]")
        print("step 1[sec]", time_1-time_0)
        print("step 2[sec]", time_2-time_1)
        print("step 3[sec]", time_3-time_2)
        print("step 4[sec]", time_4-time_3)
        print("step 5[sec]", time_5-time_4)
        print("step 6[sec]", time_6-time_5)
        

