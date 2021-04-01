import cv2
import numpy as np
import math
import scipy.linalg

import time
import matplotlib.pyplot as plt


# Rotation matrix along x/y/z axis
def get_xrot(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[  1,  0,  0],[  0, c,-s],[  0, s, c]])

def get_yrot(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[ c,  0, s],[  0,  1,  0],[-s,  0, c]])

def get_zrot(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[ c,-s,  0],[ s, c,  0],[  0,  0,  1]])


# Rotation matrix along axis 
def get_axisrot (angle, axis) :
    length = np.linalg.norm(axis)
    if length <= 0.0000001 :
        return np.identity(3)
    axis /= length
    return scipy.linalg.expm( np.cross(np.identity(3), angle*axis) )


# get rotation matrix (3x3) such that v2 = R v1 
def get_rotation_vec2vec(v1, v2) : 
    axis = np.cross(v1,v2)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 0.000001 :
        return np.identity(3, dtype=np.float32)
    # axis_norm = |v1||v2|sin(angle)
    sin_angle = axis_norm / ( np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.sqrt(1 - sin_angle**2)
    if np.dot(v1,v2) < 0: 
        cos_angle *= -1 
    angle = np.arccos( cos_angle )

    return get_axisrot(angle, axis/axis_norm)


# -------------------------------------------------------------------
# gen_2Dindex_array(height, width)
# this function gemerates two 2d index arrays (x, y)
#  x = [[0,1,2,...,w-1], [0,1,2,...,w-1], ..., [0,1,2,...,w-1]]
#  y = [[0,0,0,...,  0], [1,1,1,..., 1 ], ..., [h-1,h-1,h-1,...,h-1]]
#
def gen_index_array2d (heights, width) : 
    y = np.arange(heights).repeat(width).reshape(heights,width)
    x = np.arange(width).reshape((-1,width)).repeat(heights,axis=0)
    return y, x


# -------------------------------------------------------------------
# t_hough_circle4 (twice the accuracy of the radius)
# hough transform to search a circle in a binary image 
#  
#  memo : non accelerated codes of hough transform (t_hough_circle{,1,2,3})
#         are avaliable below
# 
def t_hough_circle4(
        img_bin,                    # target binary image 
        min_x, min_y, max_x, max_y, # rectangle area to search 
        min_cx, min_cy, min_r,      # minimum (cx, cy, r) of circle to search
        max_cx, max_cy, max_r       # maximum (cx, cy, r) of circle to search
    ) :
    vol_cx = max_cx - min_cx
    vol_cy = max_cy - min_cy
    min_r2, max_r2 = min_r*2, max_r*2
    vol_r  = max_r2  - min_r2

    if  vol_cx <= 0 or vol_cy <= 0 or vol_r <= 0 : 
        return 0, 0, 0
    vol = np.zeros((vol_r, vol_cy, vol_cx))

    y_src, x_src = gen_index_array2d(vol_cy, vol_cx)
    x_src += min_cx
    y_src += min_cy

    b = (img_bin[ min_y+1 : max_y-1 , min_x+1: max_x-1] != 0 ) & \
       ((img_bin[ min_y+2 : max_y   , min_x+1: max_x-1] == 0 ) | \
        (img_bin[ min_y   : max_y-2 , min_x+1: max_x-1] == 0 ) | \
        (img_bin[ min_y+1 : max_y-1 , min_x+2: max_x  ] == 0 ) | \
        (img_bin[ min_y+1 : max_y-1 , min_x  : max_x-2] == 0 ) )
    idx = np.array ( np.where(b) ) 
    idx[0] += min_y+1
    idx[1] += min_x+1
    for i in range(idx.shape[1]) : 
        by = idx[0,i]
        bx = idx[1,i]
        r = np.sqrt( (bx-x_src)**2 + (by-y_src)**2 ) * 2
        r1 = np.int32(r-0.5 )
        r2 = np.int32(r+0.5 )
        r3 = np.int32(r+1.5 )

        yi, xi = np.where( (min_r2 <= r1) & (r1 < max_r2) )
        ri = r1[yi, xi] - min_r2
        vol[ri,yi,xi] += 1

        yi, xi = np.where( (min_r2 <= r2) & (r2 < max_r2) )
        ri = r2[yi, xi] - min_r2
        vol[ri,yi,xi] += 2

        yi, xi = np.where( (min_r2 <= r3) & (r3 < max_r2) )
        ri = r3[yi, xi] - min_r2
        vol[ri,yi,xi] += 1

    # fine the most voted pixel
    piv_idx = np.unravel_index(np.argmax(vol), vol.shape)
    return piv_idx[2] + min_cx, piv_idx[1]+min_cy, piv_idx[0] / 2 + min_r




# -------------------------------------------------------------------
# t_cubic_supersampling_1d
# performs cubic super sampling for 1d array 
# 
# num_sample : num of sample points between src[i]~src[i+1] (integer)
# 
def t_cubic_supersampling_1d ( src, num_sample = 10 ) :
    num_points = (src.shape[0] - 1) * num_sample + 1
    trgt = np.zeros(num_points, dtype=np.float32)

    ofst = 1.0 / num_sample
    m = np.array([[0,0,0,1],
                  [1,1,1,1],
                  [8,4,2,1],
                  [27,9,3,1]], dtype=np.float32)
    mi = np.linalg.inv(m)

    # sample between src[0]~src[1]
    a,b,c,d = np.dot(mi, src[0:4] )
    for k in range(num_sample) :
        t = ofst * k
        trgt[ 0 * num_sample + k ] = a*t*t*t + b*t*t + c*t + d

    # sample between src[i+1]~src[i+2] (i=0~n-4)
    for i in range(0, src.shape[0]-3) :
        a,b,c,d = np.dot(mi, src[i:i+4] )
        for k in range(num_sample) :
            t = ofst * k + 1
            trgt[ (i+1) * num_sample + k ] = a*t*t*t + b*t*t + c*t + d

    a,b,c,d = np.dot(mi, src[-4:] )
    for k in range(num_sample) :
        t = ofst * k + 2
        trgt[ (src.shape[0]-2) * num_sample + k ] = a*t*t*t + b*t*t + c*t + d

    trgt[-1] = src[-1]
    return trgt


# -------------------------------------------------------------------
# load_video
# this function opens a video (mp4), loads all frames, and store them in a np.array()
# this returns np.array(N, W, H)
#
# memo : grayscale images loaded by cv2  are slightly different from those by ffmpeg.      
#        with this difference the cost values obtained with this python codes are 
#        different from those by C++ version 
# 
def load_video(file_path, maxnum_frames=1024) :
    cap = cv2.VideoCapture(file_path )
    ret, frame = cap.read()
    if ret == 0 :
        print("ERROR : failed to open video")
        exit()

    num_frames  = int( cap.get(cv2.CAP_PROP_FRAME_COUNT) )
    num_frames = min(num_frames, maxnum_frames)
    h, w, ch = frame.shape

    print("----load video----", num_frames, h, w)
    frames = np.zeros((num_frames, h, w), dtype = np.uint8)
    frames[0,:,:] = np.float32(cv2.cvtColor(np.flipud(frame), cv2.COLOR_BGR2GRAY))

    for i in range(1, num_frames) :
        ret, frame = cap.read()
        if ret == 0 :
            break
        frames[i,:,:] = np.float32(cv2.cvtColor(np.flipud(frame), cv2.COLOR_BGR2GRAY))
    return frames


# -------------------------------------------------------------------
# gen_morpho_operation_kernel
# this function generates "diamond"-shaped kernel 
# used for morphological operation
# 
def gen_morpho_operation_kernel( r ) :
    n = 2 * r + 1
    kernel = np.zeros((n,n), dtype=np.uint8)
    for i in range(r+1) :
        kernel[ r+i, i:n-i] = 1
        kernel[ r-i, i:n-i] = 1
    return kernel


# -------------------------------------------------------------------
# gen_mask_image
# this function generates mask image
# + 2D gaussian image (width x width ), width = 2 * mask_radi + 1
# + the area specified by "mask_angle" and "mask_size[%]"" are set as 0
#   the area is necessary to ignore the saturated pixels
#
def gen_mask_image (mask_radi, mask_angle, mask_rate) :
    mask_w = 2 * mask_radi + 1

    std_value  = 0.5 * mask_radi
    c = mask_radi
    coef_2ss   =  1.0 / ( 2 * std_value * std_value         )
    coef_2sspi =  1.0 / ( 2 * std_value * std_value * np.pi )

    y, x = gen_index_array2d(mask_w, mask_w)
    y -= c
    x -= c 
    mask = coef_2sspi * np.exp( -(x**2 + y**2) * coef_2ss )

    # set zero out side the circle
    mask[ (x**2 + y**2) > (mask_radi - 1 )**2 ] = 0

    dir_y = np.sin(mask_angle/180*np.pi)
    dir_x = np.cos(mask_angle/180*np.pi)
    t = (mask_radi * (1-mask_rate))**2
    mask[ ((x**2 + y**2) > t ) & (dir_x*x + dir_y*y > 0.000001) ] = 0
    
    return mask


# -------------------------------------------------------------------
# this function plots (points) as a 1d chart, and save it as (fname)
#
def save_chart(title1, fname, points):
    fig = plt.figure(figsize=(6.0, 6.0), dpi=100, facecolor='w', linewidth=0, edgecolor='w')
    sub = fig.add_subplot(111, title=title1, xlabel="x0", ylabel='y0' )
    sub.plot( np.arange(points.shape[0]), points)
    fig.savefig(fname)

# -------------------------------------------------------------------
# this function 
# - plots (points) as a 1d chart 
# - draw a vertical bar 
# - and save it as (fname)
#
# title1 : chart title
# fname  : file name to export
# points : y_values to plot 
# bar_x  : x_positions of vertical bars
#
def save_chart_with_bars(title1, fname, points, bar_x):
    fig = plt.figure(figsize=(8.0, 4.0), dpi=100, facecolor='w', linewidth=0, edgecolor='w')
    sub = fig.add_subplot(111, title=title1, xlabel="x0", ylabel='y0' )
    sub.plot( np.arange(points.shape[0]), points)
    
    color = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    min_y, max_y = 0.0, 1.0
    for i in range(bar_x.shape[0]) : 
        sub.plot([bar_x[i], bar_x[i]], [min_y, max_y], color[i%6], linestyle='dashed')
    
    fig.savefig(fname)


# -------------------------------------------------------------------
# this function computs ANDF  
# AMDF : Average magnitude difference function Fk = Σ | f(i) - f(i+k) |
# 
def t_AMDF(src) : 
    n = src.shape[0]
    src = np.float32(src)
    """
    amdf = np.zeros(n, dtype=np.float64)
    for k in range(n) : 
        for i in range(n) :
            if i-k < 0 :  
                amdf[k] += np.abs(src[i]-src[-(i-k)])
            else : 
                amdf[k] += np.abs(src[i]-src[ i-k ])
    """
    # acceleration above
    amdf_acc = np.zeros(n, dtype=np.float32)
    src2 = np.hstack((src[n-1:0:-1], src))
    for k in range(n) : 
        amdf_acc[k] = np.sum(np.abs(src - src2[(n-1)-k:(n-1)-k+n]))

    return amdf_acc


# -------------------------------------------------------------------
# this function applies gaussian smoothing (num_smoothing times)
#
def t_smoothing(src, num_smoothing) : 
    for i_smooth in range(num_smoothing) :
        tmp = src.copy()
        for fi in range(1, src.shape[0]-1) :
            tmp[fi] = 0.25 * src[fi-1] + 0.5 * src[fi] + 0.25 * src[fi+1]
        src = tmp.copy()
    return src


# -------------------------------------------------------------------
# normalize 1d array
def t_normalize_1d(src) : 
    minv, maxv = np.min(src), np.max(src)
    return (src - minv) / ( maxv-minv ) 


# -------------------------------------------------------------------
# generate multiple circlular templates
# templates[ 0 ] has radius self.ball_r1
# templates[n-1] has radius self.ball_r1
# templates[i]   interplates r1~r2
# 
def __gen_ball_templates(r1, r2, num_templates) :
    r = max(r1, r2)
    temp_radi = int( r * 3 / 2 )
    temp_size = 2 * temp_radi + 1
    dR  = (r2 - r1) / ( num_templates - 1)

    ball_templates = np.zeros( (num_templates, temp_size, temp_size), dtype=np.uint8 )
    for i in range(num_templates) :
        radi = int(r1 + dR * i)
        cv2.circle(ball_templates[i], (temp_radi, temp_radi), radi, (255, 255, 255), thickness=-1)
    return ball_templates


# -------------------------------------------------------------------
# t_trackball_temp_match
# this function ...
# + binarizes the video frames (frames) by background subtraction
# + adopts template matching 
# + fints valid sequences for frames (start_i, end_i)
#
def t_trackball_temp_match(
        frames,
        r1, 
        r2,
        bkgrnd_mode, 
        bkgrnd_thresh, 
        TM_STEP
    ):
    # background subtraction / morphological operation (opening)
    if   bkgrnd_mode == "mean"  : bk_img = np.average( frames         , axis=0)
    elif bkgrnd_mode == "first" : bk_img = np.average( frames[0:3,:,:], axis=0) #ave of first 3
    elif bkgrnd_mode == "last"  : bk_img = np.average( frames[-3:,:,:], axis=0) #ave of last 3

    n, h, w = frames.shape
    frames_bin = np.zeros((n, h, w), dtype=np.uint8)
    open_kernel = gen_morpho_operation_kernel(2)
    for i in range(n) :
        tmp = np.abs( frames[i,:,:] - bk_img )
        frames_bin[i, tmp > bkgrnd_thresh] = 255
        frames_bin[i] = cv2.morphologyEx( frames_bin[i], cv2.MORPH_OPEN, open_kernel)

    #3. template matching
    NUM_TEMPLATES = 5
    ball_templates = __gen_ball_templates(r1,r2,NUM_TEMPLATES)
    bestmatch_idx, bestmatch_val = 0, 1.7976931348623157e+308
    center_seq = np.zeros((n,2), dtype=np.int32  )

    for fi in range(0, n, TM_STEP) :
        idx = min( fi // ( n // NUM_TEMPLATES), NUM_TEMPLATES-1)
        temp = ball_templates[idx]
        res = cv2.matchTemplate( frames_bin[fi], temp, cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        center_seq[fi][0] = min_loc[0] + temp.shape[0]//2
        center_seq[fi][1] = min_loc[1] + temp.shape[1]//2
        if min_val < bestmatch_val :
            bestmatch_val = min_val
            bestmatch_idx = fi

    #4. extract valid template match center sequence  (by track backward/forward from bestmatch_idx )
    start_i  = bestmatch_idx
    end_i    = bestmatch_idx
    r = max(r1, r2)
    for fi in range (bestmatch_idx - TM_STEP, 0, -TM_STEP):
        if np.linalg.norm(center_seq[fi] - center_seq[fi+TM_STEP]) > r :
            break
        for i in range (1, TM_STEP):
            t = i / TM_STEP
            center_seq[fi+i] =  (1-t) * center_seq[fi] + t * center_seq[fi+TM_STEP]
        start_i   = fi

    for fi in range(bestmatch_idx + TM_STEP, n, TM_STEP) :
        if np.linalg.norm(center_seq[fi] - center_seq[fi-TM_STEP]) >  r :
            break
        for i in range(1, TM_STEP):
            t = i / TM_STEP
            center_seq[fi-TM_STEP + i] =  (1-t) * center_seq[fi-TM_STEP] + t * center_seq[fi]
        end_i += TM_STEP

    return center_seq, start_i, end_i, frames_bin


# -------------------------------------------------------------------
# t_trackball_hough
# this function adopt hough transform to binarized video frames (frames_bin)
# and returns a sequence of (cx,cy,radius)
# 
def t_trackball_hough(
        frames_bin, 
        r1, 
        r2, 
        tempmatch_cxcy,
        start_i,
        end_i  
    ):
    print("Track ball by Hough Transform...")
    r = max( r1, r2)
    n = tempmatch_cxcy.shape[0]
    hough_xyr = np.zeros((n,3), dtype=np.float32)
    
    for fi in range( start_i, end_i + 1) :
        cx, cy = tempmatch_cxcy[fi]
        hough_xyr[fi,:] = t_hough_circle4( frames_bin[fi],
                            cx - r   , cy-r   , cx + r, cy + r,
                            cx - r//2, cy-r//2, r//3,  cx + r//2, cy+r//2, r)
        if fi % 50 == 0 :
            print("compute hough transform ", fi, "/", end_i) 

    #Smoothing hough transform results
    NUM_SMOOTHING = 10
    hough_xyr[ start_i: end_i+1] = t_smoothing( hough_xyr[ start_i: end_i+1], NUM_SMOOTHING)
    return hough_xyr


# -------------------------------------------------------------------
# this function
# extracts area specified by center_xyr (cx, cy, radius)
# resize it as (clip_w, clip_w)
#
def t_generate_ballclip(
        frames,
        r1,
        r2,
        center_xyr,
        start_i, 
        end_i
    ):
    n = frames.shape[0]
    clip_r = max(r1, r2)
    clip_w = 2 * clip_r + 1
    ballclips = np.zeros((n, clip_w, clip_w), np.float32)

    for fi in range( start_i, end_i + 1 ) :
        hx, hy, hr = np.int32( center_xyr[fi] + 0.5)
        tmp = frames[fi, hy-hr:hy+hr+1, hx-hr:hx+hr+1]
        tmp = ballclips[fi] = cv2.resize(tmp, (clip_w, clip_w), interpolation = cv2.INTER_AREA)
        ballclips[fi] = tmp.copy()
    return ballclips


def t_generate_ballclip2(
        frames,
        r1,
        r2,
        center_xyr,
        start_i, 
        end_i
    ):

    n, h, w = frames.shape
    clip_r = max(r1, r2)
    clip_w = 2 * clip_r + 1
    ballclips = np.zeros((n, clip_w, clip_w), np.float32)
    
    for fi in range(start_i, end_i) : 
        frm = frames[fi,:,:]
        cx, cy, cr = center_xyr[fi]
        rate = cr / clip_r
        y, x = gen_index_array2d(clip_w, clip_w)
        y, x = y.reshape(-1), x.reshape(-1)
        py = (y - clip_r) * rate + cy - 0.5
        px = (x - clip_r) * rate + cx - 0.5
        yi = np.int32(py)
        xi = np.int32(px)
        ty = py - yi
        tx = px - xi
        a0 = (1-tx) * frm[yi  , xi] + tx * frm[yi  , xi+1]
        a1 = (1-tx) * frm[yi+1, xi] + tx * frm[yi+1, xi+1]
        a = ballclips[fi,:,:] = np.int32( np.minimum(255.0, (1-ty) * a0 + ty * a1)).reshape(clip_w, clip_w)
        """
        for y in range(-clip_r, clip_r + 1) : 
            for x in range(-clip_r, clip_r + 1) : 
                px = cx + x * rate - 0.5
                py = cy + y * rate - 0.5
                xi = int(px)
                yi = int(py)
                if yi < 0 or yi > h-1 or xi < 0 or xi > w-1: 
                    continue
                tx = px - xi
                ty = py - yi
                a0 = (1-tx) * frm[yi  , xi] + tx * frm[yi  , xi+1]
                a1 = (1-tx) * frm[yi+1, xi] + tx * frm[yi+1, xi+1]
                ballclips[fi, y+clip_r, x+clip_r] = int( min(255.0, (1-ty) * a0 + ty * a1))
        """ 
    return ballclips



# -------------------------------------------------------------------
# t_gen_ballclip_diff
# computs average image of ballclip
# and computs ballclip_diff = ballclip - average_img
# to remove shading effects
# 
# + ballclip : sequence  of ball_clipcs 
#     size -- (n_frm x w x w), 
#     ball clip is already clipped in time direction(n_frm = end_i - start_i)
#
def t_gen_ballclip_diff (
        ballclip , 
        BLUR_RATE = 0.1) :
    
    n,h,w = ballclip.shape
    ballclip_diff = np.zeros((n,h,w), dtype=np.float64)

    # Gaussian blur, calc average image,
    BLUR_RATE = 0.1
    gauss_std = h * BLUR_RATE * 0.5 
    gauss_w   = int(3*gauss_std) * 2 + 1
    for fi in range(n) :
        #ballclip_diff[fi] = cv2.GaussianBlur( ballclip[fi], (gauss_w,gauss_w), gauss_std )
        ballclip_diff[fi] = ballclip[fi]
    # subtract average image to cancel shading effects
    ballclip_ave = np.average( ballclip_diff[:,:,:], axis=0)
    for fi in range(n):
        ballclip_diff[fi] = ballclip_diff[fi] - ballclip_ave
    return ballclip_ave, ballclip_diff

    
# -------------------------------------------------------------------
# + ballclip : sequence  of ball_clipcs 
#   size -- (n_frm x w x w),  
#   type -- np.array (float32)
#   (ball_clip_diff in VideoManager, n_frm = end_i - start_i)
#
def t_estimate_spinspeed(
        ballclip  , 
        bc_mask   ,
        VIDEO_FPS ,
        MAX_RPS 
    ) :
    n, h, w = ballclip.shape
    rps_candidates = np.zeros(0)

    # calc Dij (diff between i-th and j-th ball clip)
    """
    diff_ij = np.zeros((n, n), dtype=np.float32)
    for i in range( n ) :
        for j in range(i, n ) :
            diff = np.sum(  bc_mask * (ballclip[i] - ballclip[j])**2 )
            diff_ij[i,j] = diff_ij[j,i] = diff
    """
    y, x = gen_index_array2d(n, n)
    y, x = y.reshape(-1), x.reshape(-1)
    diff_ij = np.sum( bc_mask * (ballclip[y] - ballclip[x]) ** 2, axis=(1,2) ).reshape(n,n) 
    t3 = time.time()

    # sum up Dij in diagonal direction to get vi (see 4th page of [Ijiri et al, automatic spin measurement...])
    num_vi = n * 2 // 3
    vi = np.zeros( num_vi, dtype=np.float32)
    for y in range(n // 3) :
        vi += diff_ij[ y, y : y + num_vi]

    VI_SUPERSAMPLE_RATE = 10
    VI_SMOOTHING_NUM    = 10
    vi_fine = t_cubic_supersampling_1d( vi, VI_SUPERSAMPLE_RATE)
    vi_amdf = t_AMDF(vi_fine)
    vi_amdf = t_smoothing(vi_amdf, VI_SMOOTHING_NUM)
    vi      = t_normalize_1d(vi     )
    vi_fine = t_normalize_1d(vi_fine)
    vi_amdf = t_normalize_1d(vi_amdf)

    # when the fastest spin occurs...        
    # 1 / MAX_RPS --> seconds for one revolution 
    # (1 / MAX_RPS) * VIDEO_FPS --> frames for one revolution 
    MAX_NUM_CANDIDATES = 5
    min_ids = []
    amdf_minfrm = int( ((1 / MAX_RPS) * VIDEO_FPS ) * VI_SUPERSAMPLE_RATE )
    for fi in range(amdf_minfrm+1, vi_amdf.shape[0] - 1) : 
        if vi_amdf[fi] < 0.6 and vi_amdf[fi-1] > vi_amdf[fi] and vi_amdf[fi] <=  vi_amdf[fi+1] :  
            min_ids.append(fi)
            if len (min_ids) >= MAX_NUM_CANDIDATES : 
                break

    print(min_ids)
    min_ids = np.array(min_ids, dtype=np.float32)
    # Get candidates of spin_period and RPS (notes: all candidates will be tested in the next step)
    res_spin_periods    = min_ids / VI_SUPERSAMPLE_RATE
    res_estimated_RPSs = VIDEO_FPS / res_spin_periods

    #output estimation info 
    save_chart_with_bars("vi"     , "vi_.png"     , vi     , min_ids / VI_SUPERSAMPLE_RATE)
    save_chart_with_bars("vi_fine", "vi_fine_.png", vi_fine, min_ids)
    save_chart_with_bars("vi_amdf", "vi_amdf_.png", vi_amdf, min_ids)

    print("estimated spin rate...")
    for i in range(res_spin_periods.shape[0] ) : 
        print("frame:", res_spin_periods[i]  )   
        print("Rev. Per Sec.:", res_estimated_RPSs[i]) 

    return res_spin_periods, res_estimated_RPSs





# -------------------------------------------------------------------
# calc_costs_for_axis_candidates 
# this function computes matching costs for all axis (candidate_axes)
# + ballclip : sequence  of ball_clipcs 
#     size and type -- n_frm x w x w, float32
#     shading effect is already removed (by subtracting  mean image)
# + candidate_axes : multiple candidates of rotation axis
# + frames_for_one_spin : num of frames for single revolution 
# + mask : ball clip mask (size is w x w)
# + K    : offset num of frames for matching (default is 1)
# 
def  calc_costs_for_axis_candidates(
        ballclip,
        candidate_axes,
        frames_for_one_spin, 
        bc_mask, 
        K  = 1) : 
    
    num_bc_frames, ball_w, _  = ballclip.shape
    ball_r   = ballclip.shape[1] // 2 
    w0       = 2 * np.pi / frames_for_one_spin

    # idx_x / idx_y : 2d array of pixel indices 
    idx_y_src, idx_x_src = gen_index_array2d(ball_w, ball_w)

    # map pixels onto a spehere (origin is (ball_r + 0.5,  ball_r + 0.5, 0))
    px = idx_x_src + 0.5 - (ball_r + 0.5)
    py = idx_y_src + 0.5 - (ball_r + 0.5)
    pz = ball_r **2 - px**2 - py**2 
    b_trgt_pix  = (pz >= 0) & (bc_mask>0)  # ignor pixels outside circle and mask(y,x) = 0
    pz[ pz < 0] = 0
    pz = np.sqrt(pz)
    pixel_pos = np.vstack((px.reshape(-1), py.reshape(-1), pz.reshape(-1)))

    costs_at_allaxis = np.zeros(candidate_axes.shape[0], dtype=np.float32)

    time_start = time.perf_counter()
    for axis_i, axis in enumerate(candidate_axes) : 

        # rotate all points in pixel_pos to get their distination pixel idx
        rot = get_axisrot(w0*K, axis)

        rot_pos = np.dot(rot, pixel_pos)
        rot_px = rot_pos[0,:].reshape(ball_w, ball_w) 
        rot_py = rot_pos[1,:].reshape(ball_w, ball_w) 
        rot_pz = rot_pos[2,:].reshape(ball_w, ball_w)
        idx1_x = idx_x_src.copy() 
        idx1_y = idx_y_src.copy() 
        idx2_x = np.int32(rot_px + ball_r + 0.5)
        idx2_y = np.int32(rot_py + ball_r + 0.5)

        #extract valid idx1 and idx2
        bool_trgt = b_trgt_pix & (rot_pz >= 0) & (idx2_x >= 0) & (idx2_y >= 0)
        idx1_x, idx1_y = idx1_x[bool_trgt], idx1_y[bool_trgt]
        idx2_x, idx2_y = idx2_x[bool_trgt], idx2_y[bool_trgt]

        # comput costs with fancy indexing 
        weights = bc_mask[idx1_y, idx1_x] * bc_mask[idx2_y, idx2_x] 
        sum_weight = np.sum(weights) * (num_bc_frames - K)
        tmp = weights * (  ballclip[:- K, idx1_y, idx1_x] - ballclip[K:   , idx2_y, idx2_x] )**2 
        costs_at_allaxis[axis_i] = np.sum(tmp) / sum_weight 
        if axis_i % 1000 == 0:
            print("axis estimation ", axis_i, candidate_axes.shape[0])
        """
        if axis_i == 5 : 
            print("sum_weights and cost : ", sum_weight, costs_at_allaxis[axis_i])
            # Original implementation 
            idx1 = []
            idx2 = []
            weights = [] 
            rot = get_axisrot(w0*K, axis)
            cx, cy = ball_r + 0.5, ball_r + 0.5
            for y in range(ball_w) : 
                for x in range(ball_w) : 
                    z = ball_r ** 2 - (x + 0.5-cx)**2 - (y+0.5-cy)**2
                    if z < 0 : continue
                    fp = np.dot(rot, np.array( [x+0.5-cx, y+0.5-cy, np.sqrt( z )] ))
                    xi = int( fp[0] + cx )
                    yi = int( fp[1] + cy )
                    if  fp[2] < 0 or xi < 0 or yi < 0 : continue
                    w = bc_mask[y,x] * bc_mask[yi, xi] 

                    if y == 2 and x == 24:
                        print("here !!!" , bc_mask[y,x], bc_mask[yi, xi] )

                    if w <= 0.00000001 : continue
                    idx1.append([y ,x ])
                    idx2.append([yi,xi])
                    weights.append(w)
            weightSum = 0
            energy    = 0

            for f in range(ballclip.shape[0]-K) : 
                clip1 = ballclip[f  ]
                clip2 = ballclip[f+K]
                for i in range(len(idx1)) : 
                    weightSum += weights[i]
                    energy    += weights[i] * (clip1[idx1[i][0], idx1[i][1]] - clip2[idx2[i][0], idx2[i][1]] ) * (clip1[idx1[i][0], idx1[i][1]] - clip2[idx2[i][0], idx2[i][1]] ) 
            energy /= weightSum
            for i in range(10) : 
                print(idx1[i],idx2[i],"   ") 

            print("sum_weights and cost : ", weightSum, energy)

            print("DONE!!!!", time.perf_counter() - time_start, "sec")
            """
    return costs_at_allaxis

    


# + ballclip : sequence  of ball_clipcs 
#   size -- (n_frm x w x w),  
#   type -- np.array (float32)
#   (ball_clip_diff in VideoManager, n_frm = end_i - start_i)

def t_estimate_spinaxis(
        ballclip         ,
        bc_mask          ,
        candidate_periods,
        candidate_axes 
    ) : 
    print("t_estimate_spinaxis")

    # check input 
    if candidate_periods.shape[0] <= 0 : 
        print("system fails to track ball or to estimate revolution speed")
        return

    num_candidates = candidate_periods.shape[0]
    
    period_and_diff = []

    min_diff = 1.7976931348623157e+308
    optim_axis = np.array([.0,.0,.0], dtype=np.float32)
    optim_period = 0
    res_costs_for_allaxes = np.zeros(candidate_axes.shape[0])
    for candi_i in range(candidate_periods.shape[0]) : 

        costs_for_allaxis = calc_costs_for_axis_candidates(
                ballclip,
                candidate_axes, 
                candidate_periods[candi_i], 
                bc_mask, 
                K  = 1) 
        
        idx  = np.argmin( costs_for_allaxis )
        diff = costs_for_allaxis[idx]
        axis = candidate_axes[idx]

        period_and_diff.append([candidate_periods[candi_i], diff])

        if diff < min_diff : 
            min_diff = diff 
            optim_axis = axis 
            res_costs_for_allaxes = costs_for_allaxis
            optim_period = candidate_periods[candi_i]
    
    print("---------------------------------------------")
    print("axis estimation DONE!")
    print("period and diff value : " )
    for i in range(len(period_and_diff) ) :
        print(i, period_and_diff[i])
    print("optim axis:"  , optim_axis )
    print("optim period:", optim_period)
    return optim_axis, optim_period, res_costs_for_allaxes




"""
    # Original implementation 
    idx1 = []
    idx2 = []
    weights = [] 
    rot = get_axisrot(w0*K, axis)
    cx, cy = ball_r + 0.5, ball_r + 0.5
    for y in range(ball_w) : 
        for x in range(ball_w) : 
            z = ball_r ** 2 - (x + 0.5-cx)**2 - (y+0.5-cy)**2
            if z < 0 : continue
            fp = np.dot(rot, np.array( [x+0.5-cx, y+0.5-cy, np.sqrt( z )] ))
            xi = int( fp[0] + cx )
            yi = int( fp[1] + cy )
            if  fp[2] < 0 or xi < 0 or yi < 0 : continue
            w = bc_mask[y,x] * bc_mask[yi, xi] 
            if w <= 0.00000001 : continue
            idx1.append([y ,x ])
            idx2.append([yi,xi])
            weights.append(w)
    weightSum = 0
    energy    = 0

    for f in range(ballclip.shape[0]-K) : 
        clip1 = ballclip[f  ]
        clip2 = ballclip[f+K]
        for i in range(len(idx1)) : 
            weightSum += weights[i]
            energy    += weights[i] * (clip1[idx1[i][0], idx1[i][1]] - clip2[idx2[i][0], idx2[i][1]] ) * (clip1[idx1[i][0], idx1[i][1]] - clip2[idx2[i][0], idx2[i][1]] ) 
    energy /= weightSum
    print("sum_weights and cost : ", weightSum, energy)
"""






"""
# ------------------------------------------------------------------- 
# t_hough_circle
# hough transform to search a circle in a binary image 
# 
def t_hough_circle(
        img_bin,                    # target binary image 
        min_x, min_y, max_x, max_y, # rectangle area to search 
        min_cx, min_cy, min_r,      # minimum (cx, cy, r) of circle to search
        max_cx, max_cy, max_r       # maximum (cx, cy, r) of circle to search
    ) :
    vol_cx = max_cx - min_cx
    vol_cy = max_cy - min_cy
    vol_r  = max_r  - min_r
    print(vol_cx, vol_cy, vol_r)
    if  vol_cx <= 0 or vol_cy <= 0 or vol_r <= 0 : return 0, 0, 0

    vol = np.zeros((vol_r, vol_cy, vol_cx))

    print(img_bin)
    #voting
    for by in range(min_y+1, max_y-1):
        for bx in range(min_x+1, max_x-1):
            if img_bin[by,bx] == 0: continue
            if img_bin[by,bx-1] > 0 and img_bin[by,bx+1] > 0 and img_bin[by-1,bx] > 0 and img_bin[by+1,bx] > 0 : continue

            for y in range(vol_cy) :
                for x in range(vol_cx) :
                    xx = x + min_cx
                    yy = y + min_cy
                    r = np.sqrt( (bx-xx)**2 + (by-yy)**2 )  #twice the accuracy 精度を倍に
                    r1 = int(r-0.5 )
                    r2 = int(r+0.5 )
                    r3 = int(r+1.5 )
                    if min_r <= r1 and r1 < max_r : vol[r1-min_r, y,x] += 1
                    if min_r <= r2 and r2 < max_r : vol[r2-min_r, y,x] += 2
                    if min_r <= r3 and r3 < max_r : vol[r3-min_r, y,x] += 1

    # fine the most voted pixel
    piv_idx = np.unravel_index(np.argmax(vol), vol.shape)

    return piv_idx[2] + min_cx, piv_idx[1]+min_cy, piv_idx[0] + min_r
"""


"""
# -------------------------------------------------------------------
# t_hough_circle2 (faster than t_hough_circle)
# hough transform to search a circle in a binary image 
# 
def t_hough_circle2(
        img_bin,                    # target binary image 
        min_x, min_y, max_x, max_y, # rectangle area to search 
        min_cx, min_cy, min_r,      # minimum (cx, cy, r) of circle to search
        max_cx, max_cy, max_r       # maximum (cx, cy, r) of circle to search
    ) :
    vol_cx = max_cx - min_cx
    vol_cy = max_cy - min_cy
    vol_r  = max_r  - min_r
    if  vol_cx <= 0 or vol_cy <= 0 or vol_r <= 0 : return 0, 0, 0
    vol = np.zeros((vol_r, vol_cy, vol_cx))

    #voting
    for by in range(min_y+1, max_y-1):
        for bx in range(min_x+1, max_x-1):
            if img_bin[by,bx] == 0: continue
            if img_bin[by,bx-1] > 0 and img_bin[by,bx+1] > 0 and img_bin[by-1,bx] > 0 and img_bin[by+1,bx] > 0 : continue

            #gen 2d index array of x[[0,1,2..],[0,1,2,..],..] and y[[0,0,0,...],[1,1,1,...],...]
            y, x = gen_index_array2d(vol_cy, vol_cx)

            x += min_cx
            y += min_cy
            r = np.sqrt( (bx-x)**2 + (by-y)**2 )

            r1 = np.int32(r-0.5 )
            r2 = np.int32(r+0.5 )
            r3 = np.int32(r+1.5 )

            yi, xi = np.where( (min_r <= r1) & (r1 < max_r) )
            ri = r1[yi, xi] - min_r
            vol[ri,yi,xi] += 1

            yi, xi = np.where( (min_r <= r2) & (r2 < max_r) )
            ri = r2[yi, xi] - min_r
            vol[ri,yi,xi] += 2

            yi, xi = np.where( (min_r <= r3) & (r3 < max_r) )
            ri = r3[yi, xi] - min_r
            vol[ri,yi,xi] += 1

    # fine the most voted pixel
    piv_idx = np.unravel_index(np.argmax(vol), vol.shape)
    return piv_idx[2] + min_cx, piv_idx[1]+min_cy, piv_idx[0] + min_r
"""

"""
# -------------------------------------------------------------------
# t_hough_circle3 (faster than t_hough_circle2)
# hough transform to search a circle in a binary image 
# 
def t_hough_circle3(
    img_bin,                    # target binary image 
    min_x, min_y, max_x, max_y, # rectangle area to search 
    min_cx, min_cy, min_r,      # minimum (cx, cy, r) of circle to search
    max_cx, max_cy, max_r       # maximum (cx, cy, r) of circle to search
    ):
    vol_cx = max_cx - min_cx
    vol_cy = max_cy - min_cy
    vol_r  = max_r  - min_r
    if  vol_cx <= 0 or vol_cy <= 0 or vol_r <= 0 : return 0, 0, 0
    vol = np.zeros((vol_r, vol_cy, vol_cx))

    y_src, x_src = gen_index_array2d(vol_cy, vol_cx)
    x_src += min_cx
    y_src += min_cy

    b = (img_bin[ min_y+1 : max_y-1 , min_x+1: max_x-1] != 0 ) & \
       ((img_bin[ min_y+2 : max_y   , min_x+1: max_x-1] == 0 ) | \
        (img_bin[ min_y   : max_y-2 , min_x+1: max_x-1] == 0 ) | \
        (img_bin[ min_y+1 : max_y-1 , min_x+2: max_x  ] == 0 ) | \
        (img_bin[ min_y+1 : max_y-1 , min_x  : max_x-2] == 0 ) )
    idx = np.array ( np.where(b) ) 
    idx[0] += min_y+1
    idx[1] += min_x+1
    for i in range(idx.shape[1]) : 
        by = idx[0,i]
        bx = idx[1,i]
        r = np.sqrt( (bx-x_src)**2 + (by-y_src)**2 )
        r1 = np.int32(r-0.5 )
        r2 = np.int32(r+0.5 )
        r3 = np.int32(r+1.5 )

        yi, xi = np.where( (min_r <= r1) & (r1 < max_r) )
        ri = r1[yi, xi] - min_r
        vol[ri,yi,xi] += 1

        yi, xi = np.where( (min_r <= r2) & (r2 < max_r) )
        ri = r2[yi, xi] - min_r
        vol[ri,yi,xi] += 2

        yi, xi = np.where( (min_r <= r3) & (r3 < max_r) )
        ri = r3[yi, xi] - min_r
        vol[ri,yi,xi] += 1

    # fine the most voted pixel
    piv_idx = np.unravel_index(np.argmax(vol), vol.shape)
    return piv_idx[2] + min_cx, piv_idx[1]+min_cy, piv_idx[0] + min_r
"""
