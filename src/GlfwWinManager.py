
import glfw
import numpy as np
import tutil

from OpenGL.GL import *
from OpenGL.GLU import *




def set_light_params():
    light_pos = np.array([[ 1000, 1000,-1000,1],
                          [-1000, 1000,-1000,1],
                          [ 1000,-1000,-1000,1]], dtype=np.float32)
    light_white = np.array([1.0,1.0,1.0,1.0], dtype=np.float32)
    light_gray  = np.array([0.4,0.4,0.4,1.0], dtype=np.float32)
    light_black = np.array([0.0,0.0,0.0,1.0], dtype=np.float32)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)
    glEnable(GL_LIGHT2)
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos[0])
    glLightfv(GL_LIGHT1, GL_POSITION, light_pos[1])
    glLightfv(GL_LIGHT2, GL_POSITION, light_pos[2])
    glLightfv(GL_LIGHT0, GL_AMBIENT , light_white )
    glLightfv(GL_LIGHT1, GL_AMBIENT , light_black )
    glLightfv(GL_LIGHT2, GL_AMBIENT , light_black )
    glLightfv(GL_LIGHT0, GL_DIFFUSE , light_white )
    glLightfv(GL_LIGHT1, GL_DIFFUSE , light_gray  )
    glLightfv(GL_LIGHT2, GL_DIFFUSE , light_gray  )
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_white )
    glLightfv(GL_LIGHT1, GL_SPECULAR, light_white )
    glLightfv(GL_LIGHT2, GL_SPECULAR, light_white )




# Class GlfwWinManager
# this class
# + generates & manages glfw_window
# + manages mouse events
#    following call back functions should be set at constractor
#    func_Ldown(x,y,glfw_win_manager), func_Lup(x,y,glfw_win_manager),
#    func_Rdown(x,y,glfw_win_manager), func_Rup(x,y,glfw_win_manager),
#    func_Mdown(x,y,glfw_win_manager), func_Mup(x,y,glfw_win_manager),
#    func_mouse_move(x,y,glfw_win_manager),
#    func_draw_scene(glfw_win_manager)
#    memo (x, y): mouse position,
#         window: instance of GlfwMainWindow
#    func_on_keydown(key, glfw_win_manager), func_on_keydown(key, glfw_win_manager)
#
class GlfwWinManager:

    def __init__(
            self,
            window_title,
            window_size,
            window_position,
            func_Ldown, func_Lup,
            func_Rdown, func_Rup,
            func_Mdown, func_Mup,
            func_mouse_move,
            func_mouse_wheel,
            func_draw_scene,
            func_on_keydown, func_on_keyup, func_on_keykeep
        ):

        self.window = glfw.create_window(window_size[0], window_size[1], window_title, None, None)
        self.cam_pos = np.array([0.0, 0.0, 1000.0], dtype=np.float32)
        self.cam_up  = np.array([0.0, 1.0,    0.0], dtype=np.float32)
        self.cam_cnt = np.array([0.0, 0.0,    0.0], dtype=np.float32)
        self.b_rendering = False
        self.viewscale = 1000
        self.clearcolor = np.array([0.2, 0.2, 0.2, 1.0])

        if not self.window:
            glfw.terminate()
            raise RuntimeError('Could not create an window')

        self.func_Ldown, self.func_Lup = func_Ldown, func_Lup
        self.func_Rdown, self.func_Rup = func_Rdown, func_Rup
        self.func_Mdown, self.func_Mup = func_Mdown, func_Mup
        self.func_mouse_move  = func_mouse_move
        self.func_draw_scene  = func_draw_scene
        self.func_on_keydown  = func_on_keydown
        self.func_on_keyup    = func_on_keyup
        self.func_on_keykeep  = func_on_keykeep
        self.func_mouse_wheel = func_mouse_wheel

        #set callback functions
        glfw.set_cursor_pos_callback    (self.window, self.cursor_pos    )
        glfw.set_cursor_enter_callback  (self.window, self.cursor_enter  )
        glfw.set_mouse_button_callback  (self.window, self.mouse_button  )
        glfw.set_scroll_callback        (self.window, self.mouse_wheel   )
        glfw.set_window_refresh_callback(self.window, self.window_refresh)
        glfw.set_key_callback           (self.window, self.func_keyboard)
        glfw.set_window_pos             (self.window, window_position[0], window_position[1])

        #call display from here (TODO check! is this necessary?)
        glfw.make_context_current(self.window)
        self.display()   # necessary only on Windows

    def set_campos(self, x,y):
        self.cam_pos[0] = x
        self.cam_pos[1] = y
        self.cam_cnt[0] = x
        self.cam_cnt[1] = y

    def get_viewscale(self):
        return self.viewscale

    def set_viewscale(self, scale):
        self.viewscale = scale

    def cursor_pos(self, window, xpos, ypos):
        self.func_mouse_move( (xpos, ypos), self)

    def cursor_enter(self, window, entered):
        #print( 'cursor_enter:',entered, id(self.window), id(window))
        pass

    def func_keyboard(self, window, key, scancode, action, mods):

        if action == glfw.PRESS:
            self.func_on_keydown( key, self)
        elif action == glfw.RELEASE:
            self.func_on_keyup  ( key, self)
        elif action == glfw.REPEAT:
            self.func_on_keykeep( key, self)

    def mouse_button(self, window, button, action, mods):
        point = glfw.get_cursor_pos(window) #point : 2dtuple
        if   ( button == 0 and action == 1) : self.func_Ldown( point, self)
        elif ( button == 1 and action == 1) : self.func_Rdown( point, self)
        elif ( button == 2 and action == 1) : self.func_Mdown( point, self)
        elif ( button == 0 and action == 0) : self.func_Lup  ( point, self)
        elif ( button == 1 and action == 0) : self.func_Rup  ( point, self)
        elif ( button == 2 and action == 0) : self.func_Mup  ( point, self)

    def mouse_wheel(self, window, xoffset, yoffset) :
        self.func_mouse_wheel(yoffset, self)


    def window_should_close(self):
        return glfw.window_should_close(self.window)


    def wait_events_timeout(self):
        glfw.make_context_current(self.window)
        glfw.wait_events_timeout(1e-3)


    def window_refresh(self, window):
        self.display()


    def __draw_begin(self):
        if self.b_rendering : return
        self.b_rendering = True
        glfw.make_context_current(self.window)

        #set viewport and projection matrix
        view_w, view_h = glfw.get_window_size(self.window)
        glViewport(0, 0, view_w, view_h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        x_y, y_x = view_w/view_h, view_h/view_w
        r = self.viewscale
        view_near, view_far = 0.1, 3000.0
        if  view_w > view_h : glOrtho( -x_y * r, x_y * r, -    r,       r, view_near, view_far)
        else		        : glOrtho(       -r,       r, -y_x*r, y_x * r, view_near, view_far)

        #set ModelView matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt( self.cam_pos[0], self.cam_pos[1], self.cam_pos[2],
                   self.cam_cnt[0], self.cam_cnt[1], self.cam_cnt[2],
                   self.cam_up [0], self.cam_up [1], self.cam_up [2])
        glClearColor( self.clearcolor[0], self.clearcolor[1],
                      self.clearcolor[2], self.clearcolor[3])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_ACCUM_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        set_light_params()


    def __draw_end(self):
        glfw.swap_buffers(self.window)
        glfw.make_context_current(None)
        self.b_rendering = False



    def display(self):
        self.__draw_begin()

        glDisable(GL_LIGHTING)
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3d ( 1.0, 0.0, 0.0)
        glVertex3d( 0.0, 0.0, 0.0)
        glVertex3d( 10.0, 0.0, 0.0)
        glColor3d ( 0.0, 1.0, 0.0)
        glVertex3d( 0.0, 0.0, 0.0)
        glVertex3d( 0.0, 10.0, 0.0)
        glColor3d ( 0.0, 0.0, 1.0)
        glVertex3d( 0.0, 0.0, 0.0)
        glVertex3d( 0.0, 0.0, 10.0)

        glEnd()
        self.func_draw_scene(self)
        self.__draw_end()


    def camera_translate (self, dx, dy) :
        cx, cy = glfw.get_window_size(self.window)
        pixel_width = 0.01
        if cx > cy and cy > 0 : pixel_width = 2 * self.viewscale / cy
        else                  : pixel_width = 2 * self.viewscale / cx
        x_dir = np.cross(self.cam_pos - self.cam_cnt, self.cam_up)
        x_dir /= np.linalg.norm(x_dir)
        trans = (pixel_width * dx) * x_dir + (pixel_width * dy) * self.cam_up
        self.cam_pos += trans
        self.cam_cnt += trans

    def camera_zoom(self, dx, dy) :
        self.viewscale *= (1.0 - dy*0.001)
        if self.viewscale < 0.01 : self.viewscale = 0.01


    def __unproject(self, x, y, z):
        if not self.b_rendering :
            glfw.make_context_current(self.window)

        model_mat = glGetFloatv(GL_MODELVIEW_MATRIX)
        proj_mat  = glGetFloatv(GL_PROJECTION_MATRIX)
        vp = glGetIntegerv(GL_VIEWPORT)
        pos = gluUnProject( x, vp[3] - y, z, model_mat.astype('d'), proj_mat.astype('d'), vp)

        if not self.b_rendering :
            glfw.make_context_current(None)
        return np.array(pos, dtype=np.float32)


    def get_cursor_ray(self, point):
        pos1 = self.__unproject( point[0], point[1], 0.01)
        pos2 = self.__unproject( point[0], point[1], 0.2 )
        ray_dir = pos2 - pos1
        ray_dir /= np.linalg.norm(ray_dir)
        return pos1, ray_dir
