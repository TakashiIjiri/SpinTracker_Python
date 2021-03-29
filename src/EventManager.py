from VideoManager import *

import glfw
import numpy as np
import cv2

from OpenGL.GL import *
from OpenGL.GLU import *
from VideoManager import *

def draw_rect (rect, color3d) :
    x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]
    glDisable(GL_LIGHTING)
    glColor3dv(color3d)
    glLineWidth(2)
    glBegin(GL_LINE_STRIP)
    glVertex3f( x0, y0, 0.1)
    glVertex3f( x1, y0, 0.1)
    glVertex3f( x1, y1, 0.1)
    glVertex3f( x0, y1, 0.1)
    glVertex3f( x0, y0, 0.1)
    glEnd()


def draw_circle (cx, cy, radi, color3d) :
    glDisable(GL_LIGHTING)
    glColor3dv(color3d)
    glLineWidth(2)
    glBegin(GL_LINE_STRIP)
    for i in range(20) : 
        t = np.pi * 2 * i / 19
        glVertex3d(cx + radi * np.cos(t), cy + radi * np.sin(t), 0.1)
    glEnd()


def draw_spinaxis_sphere( size, img, rotmat, axis_dir) :

    glPushMatrix()    
    m = np.identity(4, dtype=np.float32)
    m[0:3,0:3] = rotmat[0:3,0:3]
    glMultMatrixf(m.transpose())

    # draw rectangle
    draw_rect ( np.array([-size,-size,size,size]), np.array([0,0,1])) 

    # draw textured sphere
    glDisable(GL_LIGHTING)
    glColor3d(1,1,1)
    scale_xyz = np.array([size,size,size])
    sphere    = VideoManager.get_inst().get_sphere_model()
    tmesh.t_draw_textured_model(scale_xyz, sphere, img  )
    
    # draw spin axis 
    a_color = np.array([[.1,.5,.8,.5], [.2,.2,1.,.5], [1,1,1,.5], [64,0,0,0.5] ], dtype=np.float32)
    glEnable(GL_LIGHTING)
    glPushMatrix()
    axis_dir /= np.linalg.norm(axis_dir)
    rotmat = tutil.get_rotation_vec2vec(np.array([0.,1.,0.]), axis_dir )
    m[0:3,0:3] = rotmat[0:3,0:3]
    glMultMatrixf(m.transpose())
    tmesh.t_draw_cylinder(size*1.5, size*0.05, a_color[0], a_color[1], a_color[2], a_color[3])
    glPopMatrix()

    # draw arrow (not yet implemented)
    glPopMatrix()




def draw_cost_sphere( size, rotmat, axis_dir, costs) :
    sphere = VideoManager.get_inst().get_sphere_model()
    axes   = sphere.verts
    if axes.shape[0] != costs.shape[0] : 
        return 

    #if costs.shape[0] != 0 and costs[0] == 0 : 
    #    return 

    glPushMatrix()    
    m = np.identity(4, dtype=np.float32)
    m[0:3,0:3] = rotmat[0:3,0:3]
    glMultMatrixf(m.transpose())

    # draw rectangle
    draw_rect ( np.array([-size,-size,size,size]), np.array([0,0,1])) 

    # draw costs sphere
    glDisable(GL_LIGHTING)
    glColor3d(1,1,1)
    glPushMatrix()
    glScaled(size,size,size)
    glPointSize(10)
    glDisable(GL_LIGHTING)
    glBegin(GL_POINTS)
    for i in range(axes.shape[0]) : 
        glColor3d(costs[i],0,0)
        glVertex3fv(axes[i])
    glEnd()       
    glPopMatrix()

    # draw spin axis 
    a_color = np.array([[.1,.5,.8,.5], [.2,.2,1.,.5], [1,1,1,.5], [64,0,0,0.5] ], dtype=np.float32)
    glEnable(GL_LIGHTING)
    glPushMatrix()
    axis_dir /= np.linalg.norm(axis_dir)
    rotmat = tutil.get_rotation_vec2vec(np.array([0.,1.,0.]), axis_dir )
    m[0:3,0:3] = rotmat[0:3,0:3]
    glMultMatrixf(m.transpose())
    tmesh.t_draw_cylinder(size*1.5, size*0.05, a_color[0], a_color[1], a_color[2], a_color[3])
    glPopMatrix()

    # draw arrow (not yet implemented)
    glPopMatrix()









BC_VIS_SIZE = 300
BALL_VIS_SIZE = 300




# class EventManager
# this class manages mouse events
# このクラスにマウスイベント処理・描画処理を集約


class EventManager:

    def __init__(self):
        self.b_Lbtn = False
        self.b_Rbtn = False
        self.b_Mbtn = False
        self.pre_pos, self.init_pos = (0,0), (0,0)
        self.b_set_roirect = False
        self.b_set_zoomwin = False
        self.b_drag_ballvis = False
        self.ballvis_rot   = np.identity(3, dtype=np.float32)


    def func_Ldown(self, point, glfw_manager) :
        self.b_Lbtn = True
        self.pre_pos = self.init_pos = point

        if glfw.get_key(glfw_manager.window, glfw.KEY_LEFT_SHIFT) or \
            glfw.get_key(glfw_manager.window, glfw.KEY_RIGHT_SHIFT): 
            self.b_set_roirect = True
            return
        elif glfw.get_key(glfw_manager.window, glfw.KEY_SPACE) : 
            self.b_set_zoomwin = True
            return

        #check dragging ball_vis
        ray_p, ray_d = glfw_manager.get_cursor_ray(point)
        h, w = VideoManager.get_inst().get_frame_size()
        x0, y0, size = w + BC_VIS_SIZE, 0, BALL_VIS_SIZE * 6
        if x0 < ray_p[0] < x0 + size and y0 < ray_p[1] < y0 + size : 
            self.b_drag_ballvis = True


    def func_Lup(self, point, glfw_manager):
        self.b_Lbtn = False
        self.b_set_roirect = False
        self.b_set_zoomwin = False
        self.b_drag_ballvis = False

    def func_Rdown(self, point, glfw_manager):
        self.pre_pos = self.init_pos = point
        self.b_Rbtn = True

    def func_Rup(self, point, glfw_manager):
        self.b_Rbtn = False

    def func_Mdown(self, point, glfw_manager):
        self.pre_pos = self.init_pos = point
        self.b_Mbtn = True

    def func_Mup(self, point,glfw_manager):
        self.b_Mbtn = False


    def func_on_keydown(self, key, glfw_manager):
        if key == 262 : 
            VideoManager.get_inst().increment_current_frame_idx()
            glfw_manager.display()
        elif key == 263 : 
            VideoManager.get_inst().decrement_current_frame_idx()
            glfw_manager.display()

    def func_on_keykeep(self, key, glfw_manager):
        if key == 262 : 
            VideoManager.get_inst().increment_current_frame_idx()
            glfw_manager.display()
        elif key == 263 : 
            VideoManager.get_inst().decrement_current_frame_idx()
            glfw_manager.display()

    def func_on_keyup(self, key, glfw_manager):
        print("func_on_keyup", key)


    def func_mouse_move(self, point, glfw_manager):
        if not (self.b_Lbtn or self.b_Rbtn or self.b_Mbtn) :
            return

        if self.b_set_roirect : 
            p0, d0 = glfw_manager.get_cursor_ray(self.init_pos)
            p1, d1 = glfw_manager.get_cursor_ray(point)
            VideoManager.get_inst().set_roi_rect([p0[0], p0[1], p1[0], p1[1]])
        elif self.b_set_zoomwin : 
            p1, d1 = glfw_manager.get_cursor_ray(point)
            VideoManager.get_inst().set_zoom_win_center((p1[0], p1[1]))
        elif self.b_drag_ballvis:
            dx, dy = point[0] - self.pre_pos[0], point[1] - self.pre_pos[1] 
            Rx = tutil.get_xrot( 0.01 * dy )
            Ry = tutil.get_yrot( 0.01 * dx )
            self.ballvis_rot = Rx.dot( Ry.dot(self.ballvis_rot) )
        else : 
            dx, dy = point[0] - self.pre_pos[0], point[1] - self.pre_pos[1] 
            if self.b_Lbtn : glfw_manager.camera_translate(dx, dy)
            if self.b_Mbtn : glfw_manager.camera_zoom(dx, dy)
            if self.b_Rbtn : glfw_manager.camera_translate(dx, dy)
        self.pre_pos = point
        glfw_manager.display()



    def func_draw_scene(self, glfw_manager):
        frame_idx  = VideoManager.get_inst().get_current_frame_idx()
        frame      = VideoManager.get_inst().get_frame_uint8(frame_idx)
        frame_roi  = VideoManager.get_inst().get_roi_frame_bin_uint8(frame_idx)

        ball_clip      = VideoManager.get_inst().get_ball_clip_uint8(frame_idx)
        ball_clip_diff = VideoManager.get_inst().get_ball_clip_diff_uint8(frame_idx)
        ball_clip_mask = VideoManager.get_inst().get_ball_clip_mask_uint8()
        ball_clip_ave  = VideoManager.get_inst().get_ball_clip_average_uint8()

        rect_roi   = np.int32( VideoManager.get_inst().get_roi_rect() )
        rect_zoom  = np.int32( VideoManager.get_inst().get_zoom_win_rect() )
        rect_tempmatch = VideoManager.get_inst().get_tempmatch_rect(frame_idx)
        cxcyr = VideoManager.get_inst().get_hough_cxcyr(frame_idx)

        spin_axis = VideoManager.get_inst().get_spin_axis()

        # render mainvideo, RoiRect, ZoomWinRect
        h, w = frame.shape
        tmesh.t_draw_image(w, h, frame)
        draw_rect(rect_roi , np.array([1.0,1.0,0.0]))
        draw_rect(rect_zoom, np.array([1.0, 0.3, 0.3]))
        rx,ry = rect_roi[0], rect_roi[1]
        draw_rect(rect_tempmatch + np.array([rx,ry,rx,ry]), np.array([0.0, 1.0, 1.0]))
        draw_circle( rx+ cxcyr[0], ry+ cxcyr[1], cxcyr[2], np.array([1.0, 0.0, 0.0]))        
        


        #draw zoom window
        glTranslate(0, h, 0)
        c = 5
        zoom_rect_img = frame[ rect_zoom[1]: rect_zoom[3]+1, rect_zoom[0]:rect_zoom[2]+1]
        zoom_w = zoom_rect_img.shape[1] * c
        zoom_h = zoom_rect_img.shape[1] * c
        r1, r2 = VideoManager.get_inst().get_ball_radius()
        tmesh.t_draw_image( zoom_w, zoom_h, zoom_rect_img)
        draw_circle( zoom_w / 2, zoom_w / 2, r1 * c,  np.array([1.0, 0.3, 0.3]))
        draw_circle( zoom_w / 2, zoom_w / 2, r2 * c,  np.array([1.0, 0.3, 0.3]))
        glTranslate(0,-h,0)

        #draw roi_frames_bin
        glTranslate(-frame_roi.shape[0],0,0)
        tmesh.t_draw_image(frame_roi.shape[1], frame_roi.shape[0], frame_roi)
        draw_rect(rect_tempmatch, np.array([0.0, 1.0, 1.0]) )
        draw_circle( cxcyr[0], cxcyr[1], cxcyr[2], np.array([1.0, 0.0, 0.0]))        
        glTranslate(frame_roi.shape[0],0,0)

        #ball clip 
        glTranslate( w, 0, 0)
        tmesh.t_draw_image( BC_VIS_SIZE, BC_VIS_SIZE, ball_clip)
        glTranslate( 0, BC_VIS_SIZE, 0)
        tmesh.t_draw_image( BC_VIS_SIZE, BC_VIS_SIZE, ball_clip_diff, color="rgb")
        glTranslate( 0, BC_VIS_SIZE, 0)
        tmesh.t_draw_image( BC_VIS_SIZE, BC_VIS_SIZE, ball_clip_ave)
        glTranslate( 0, BC_VIS_SIZE, 0)
        tmesh.t_draw_image( BC_VIS_SIZE, BC_VIS_SIZE, ball_clip_mask)
        glTranslate(-w,-BC_VIS_SIZE*3, 0)
        
        # ball 3d vis 
        ofst = w + BC_VIS_SIZE
        glTranslate( ofst + BALL_VIS_SIZE, BALL_VIS_SIZE, 0)
        draw_spinaxis_sphere(BALL_VIS_SIZE, ball_clip, self.ballvis_rot, spin_axis )
        glTranslate(-ofst - BALL_VIS_SIZE,-BALL_VIS_SIZE, 0)

        glTranslate( ofst + BALL_VIS_SIZE, 4*BALL_VIS_SIZE, 0)
        costs = VideoManager.get_inst().get_axis_estim_costs()
        draw_cost_sphere(BALL_VIS_SIZE, self.ballvis_rot, spin_axis,  costs)
        glTranslate(-ofst - BALL_VIS_SIZE,-4*BALL_VIS_SIZE, 0)

        
