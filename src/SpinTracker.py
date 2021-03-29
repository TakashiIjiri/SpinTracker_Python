from VideoManager import *

import glfw
import numpy as np
import time

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

from OpenGL.GL import *
from OpenGL.GLU import *
import GlfwWinManager
from EventManager import *

import cv2







class MainDialog(ttk.Frame):

    def __init__(self, root_):
        super().__init__(root_)

        #initialize glfw window
        self.manager = EventManager()
        self.glfw_manager = GlfwWinManager.GlfwWinManager(
            "Main Window", [800, 600], [100,100],
            self.manager.func_Ldown, self.manager.func_Lup,
            self.manager.func_Rdown, self.manager.func_Rup,
            self.manager.func_Mdown, self.manager.func_Mup,
            self.manager.func_mouse_move, self.manager.func_draw_scene, 
            self.manager.func_on_keydown, self.manager.func_on_keyup, self.manager.func_on_keykeep) 

        #initialize tkinter Frame
        self.root = root_
        self.pack()

        quit_btn = ttk.Button(self,text="Quit",command = self.quit_spintracker )
        quit_btn.pack(side="top", anchor=tk.E)

        #----frame_idx --- (slider) 
        slider_frame = tk.Frame(self, pady = 3)
        slider_frame.pack(side="top")
        num_frames = VideoManager.get_inst().num_frames()
        self.slider_val = tk.IntVar()
        self.slider = ttk.Scale(slider_frame, variable=self.slider_val, takefocus=1, 
                                length=280, from_=0, to=num_frames-1, command=self.slider_changed )
        self.slider.pack(side="left")
        self.slider_label = ttk.Label(slider_frame, text="0000")
        self.slider_label.pack(side="right")

        #----fps---- (radio button)
        fps_frame = tk.Frame(self, pady = 3)
        fps_frame.pack(side="top", anchor=tk.W)
        fps_label = ttk.Label(fps_frame, text="FPS of video:")

        self.fps_mode = tk.StringVar(None, '480')
        rb1 = ttk.Radiobutton( fps_frame, text= '300', value= '300', variable=self.fps_mode)
        rb2 = ttk.Radiobutton( fps_frame, text= '480', value= '480', variable=self.fps_mode)
        rb3 = ttk.Radiobutton( fps_frame, text= '960', value= '960', variable=self.fps_mode)
        rb4 = ttk.Radiobutton( fps_frame, text='1000', value='1000', variable=self.fps_mode)
        fps_label.pack(side="left")
        rb1.pack(side="left")
        rb2.pack(side="left")
        rb3.pack(side="left")
        rb4.pack(side="left")

        #----ball radius ---- (spin control memo:http://www.nct9.ne.jp/m_hiroi/light/py3tk05.html)
        ballrad_frame = tk.Frame(self, pady = 5)
        ballrad_frame.pack(side="top", anchor=tk.W)
        self.radi_release = tk.StringVar(None, "28")
        self.radi_catch   = tk.StringVar(None, "16")
        VideoManager.get_inst().set_ball_radius( 28, 16 )
        spinbox_release   = ttk.Spinbox( ballrad_frame, from_ = 5, to = 40, increment = 1, width = 5, state='readonly',
                                        textvariable=self.radi_release, command=self.spin_changed)
        spinbox_catch     = ttk.Spinbox( ballrad_frame, from_ = 5, to = 40, increment = 1, width = 5, state='readonly',
                                        textvariable=self.radi_catch, command=self.spin_changed)
        radi_label1 = ttk.Label(ballrad_frame, text="Radius at Release : ")
        radi_label2 = ttk.Label(ballrad_frame, text="Radius at   Catch : ")
        radi_label1    .grid(row=0, column=0)
        spinbox_release.grid(row=0, column=1)
        radi_label2    .grid(row=1, column=0)
        spinbox_catch  .grid(row=1, column=1)

        #-----background subtraction ----
        backsubt_frame  = tk.Frame(self,  pady = 5)
        backsubt_frame.pack(side="top", anchor=tk.W)
        self.backsubt_mode   = tk.StringVar(None, 'mean')
        self.backsubt_thresh = tk.StringVar(None, "12")
        backsubt_label1 = ttk.Label(backsubt_frame, text="Background subtraction, mode and threshold")
        backsubt_rb1 = ttk.Radiobutton( backsubt_frame, text= 'mean' , value= 'mean' , variable=self.backsubt_mode)
        backsubt_rb2 = ttk.Radiobutton( backsubt_frame, text= 'first', value= 'first', variable=self.backsubt_mode)
        backsubt_rb3 = ttk.Radiobutton( backsubt_frame, text= 'last' , value= 'last' , variable=self.backsubt_mode)
        backsubt_thresh  = ttk.Spinbox( backsubt_frame, from_ = 2, to = 100, increment = 1, width = 5, state='readonly',
                                        textvariable=self.backsubt_thresh, command=self.spin_changed)
        backsubt_label1.pack(side = "top")
        backsubt_rb1.pack(side="left")
        backsubt_rb2.pack(side="left")
        backsubt_rb3.pack(side="left")
        backsubt_thresh.pack(side="left")

        #----- Mask angle and size 
        mask_frame = tk.Frame(self, pady=5)
        mask_frame.pack(side="top", anchor=tk.W)
        mask_label1 = ttk.Label(mask_frame, text="mask angle [deg]: ")
        mask_label2 = ttk.Label(mask_frame, text="mask size  [-%-]: ")
        self.mask_angle_val  = tk.StringVar(None, "90")
        self.mask_rate_val   = tk.StringVar(None, "100")
        mask_angle = ttk.Spinbox( mask_frame, from_ = -180, to = 180, increment = 1, width = 5, state='readonly',
                                    textvariable=self.mask_angle_val, command=self.spin_mask_changed)
        mask_size  = ttk.Spinbox( mask_frame, from_ = 0, to = 100, increment = 5, width = 5, state='readonly',
                                    textvariable=self.mask_rate_val, command=self.spin_mask_changed)
        mask_label1.grid(row=0, column=0)
        mask_angle .grid(row=0, column=1)
        mask_label2.grid(row=1, column=0)
        mask_size  .grid(row=1, column=1)

        #----- Morph. Ope. / template match interval ----
        miscs_frame  = tk.Frame(self, pady = 5)
        miscs_frame.pack(side="top", anchor=tk.W)
        miscs_label1 = ttk.Label(miscs_frame, text="Morph. Ope. Size.      : ")
        miscs_label2 = ttk.Label(miscs_frame, text="Templ. Match. interval : ")
        self.miscs_morpho     = tk.StringVar(None, "2")
        self.miscs_tminterval = tk.StringVar(None, "2")
        miscs_morpho  = ttk.Spinbox( miscs_frame, from_ = 1, to = 10, increment = 1, width = 5, state='readonly',
                                    textvariable=self.miscs_morpho, command=self.spin_changed)
        miscs_tminter = ttk.Spinbox( miscs_frame, from_ = 1, to = 10, increment = 1, width = 5, state='readonly',
                                    textvariable=self.miscs_tminterval, command=self.spin_changed)
        miscs_label1 .grid(row=0, column=0)
        miscs_morpho .grid(row=0, column=1)
        miscs_label2 .grid(row=1, column=0)
        miscs_tminter.grid(row=1, column=1)

        #---- Buttons ------
        btn_frame  = tk.Frame(self, pady = 3)
        btn_frame.pack(side="top", anchor=tk.W)
        btn_import_conf  = ttk.Button(btn_frame,text="Inport config", command = self.import_config )
        btn_export_conf  = ttk.Button(btn_frame,text="Export config", command = self.export_config )
        btn_run_tracking = ttk.Button(btn_frame,text="Run Tracking" , command = self.run_tracking  )
        btn_import_conf.pack (side="left", anchor=tk.W)
        btn_export_conf.pack (side="left", anchor=tk.W)
        btn_run_tracking.pack(side="left", anchor=tk.W)

        # set focus on slider 
        self.slider.focus_set()

        VideoManager.get_inst().update_ballclip_mask( 
            int(self.radi_release.get()), 
            int(self.radi_catch.get()  ), 
            int(self.mask_angle_val.get()), 
            float(self.mask_rate_val.get()) / 100.0 )


    def quit_spintracker(self):
        exit()


    def slider_changed(self, e):
        fi = self.slider_val.get()
        tmp = str(fi)
        if   fi < 10 : tmp = "000" + tmp 
        elif fi < 100 : tmp = "00" + tmp 
        elif fi < 1000 : tmp = "0" + tmp 
        self.slider_label['text'] = tmp
        VideoManager.get_inst().set_current_frame_idx(fi)
        self.glfw_manager.display()
        self.slider.focus_set()

    
    # This function monitors modification of the current video idx 
    # note : glfwのイベントからtkinterのwidgetを変更すると
    #「Fatal Python error: PyEval_RestoreThread: NULL tstate」 が出る可能性あり
    # tkinterの widgetはtkinterのみから編集する
    # note : "Fatal Python error: PyEval_RestoreThread: NULL tstate"may occur, 
    # if tkinter widgets are modified from the thread of glfw
    # tkinter widget should be modified from events of tkinter
    def poling_video_frame_update(self):
        fi_1 = VideoManager.get_inst().get_current_frame_idx()
        fi_2 = self.slider_val.get()
        if fi_1 == fi_2 : return
        self.slider_val.set(fi_1)
        tmp = str(fi_1)
        if   fi_1 < 10 : tmp = "000" + tmp 
        elif fi_1 < 100 : tmp = "00" + tmp 
        elif fi_1 < 1000 : tmp = "0" + tmp 
        self.slider_label['text'] = tmp


    def spin_changed(self):
        print("radi_release", self.radi_release.get())
        print("radi_catch  ", self.radi_catch.get())
        print("bk thresh   ", self.backsubt_thresh.get())
        VideoManager.get_inst().set_ball_radius( int(self.radi_release.get()), int(self.radi_catch.get()) )
        self.glfw_manager.display()

    def spin_mask_changed(self):
        VideoManager.get_inst().update_ballclip_mask( 
            int(self.radi_release.get()), 
            int(self.radi_catch.get()  ), 
            int(self.mask_angle_val.get()), 
            float(self.mask_rate_val.get()) / 100.0 )
        self.glfw_manager.display()


    def export_config(self):
        pass


    def import_config(self):
        pass


    def run_tracking(self):
        
        VideoManager.get_inst().run_taracking_and_spinestimation(
            radius_release = int(self.radi_release     .get() ), 
            radius_catch   = int(self.radi_catch       .get() ), 
            bkgrnd_mode    = "mean", #TODO
            bkgrnd_thresh  = int(self.backsubt_thresh  .get() ),
            morpho_size    = int( self.miscs_morpho    .get() ), 
            tempmatch_step = int( self.miscs_tminterval.get() ) ,
            mask_angle     = int( self.mask_angle_val  .get() ), 
            mask_rate      = float(self.mask_rate_val  .get()) / 100,
            video_fps      = int(self.fps_mode.get())
        )

        self.glfw_manager.display()




# + 動画を読み込み(file dialogにより指定) OK
# + OK dialogに GUI 追加 
#   - SetRect v.s. Zoom window (Shift+drag --> SegRect) 
#   - OK --- SeekBar
#   - OK --- fps 300/480...
#   - OK Ball Radius (release/catch)
#   - OK BK subt threash / Morpho Ope size / TempMatch interval
#   - OK BK mode (mean image / first image / last image)
# + OK dialogに各種ボタンを配置
#   - Export Config File (txt)
#   - Import Config File (txt)
#   - perform tracking & estimation 
# TODO import/export config file
# + OK videoを表示
# + OK ROI を指定可能に
# + OK Trackingして結果を表示
# DONE Directional mask  
# DONE  and its visualization
# DONE key board input to proceed the frames
# DONE  spin speed candidate estimation 
# DONE its visualization (matplotlib)
# DONE spin axis estimation 
# DONE its visualization (3D viewer)
# NOTE : resulting costs are different from the C++ version because 
# the mp4 loaders (ffmpeg in C++ and OpenCV in python) returns 
# slightly diffent grayscale images.

def main():
    # initialize VideoManager (Singleton)
    VideoManager.get_inst()
    print(VideoManager.get_inst().num_frames())

    #prepare glfw
    if not glfw.init():
        raise RuntimeError("Fails to initialize glfw")

    app  = tk.Tk()
    app.title("SpinTracker Parameter")
    app.geometry("330x350")
    dialog = MainDialog(app)

    # note: when use tk.mainloop(), it is difficult to track errors 
    # that occur in call_back functions of glfw
    """
    note1: 下でループすると、tkが表示されない
    while not ( dialog.glfw_manager.window_should_close()):
        dialog.glfw_manager.wait_events_timeout()

    note2 : 下でループすると、glfwで起きたエラーがトラックできない
    tk.mainloop()
    """

    def custom_main_loop():
        if dialog.glfw_manager.window_should_close() :
            exit()
        dialog.poling_video_frame_update()
        dialog.glfw_manager.wait_events_timeout()
        app.after(33, custom_main_loop)

    app.after(10, custom_main_loop)
    tk.mainloop()

    glfw.terminate()

if __name__ == "__main__":
    main()