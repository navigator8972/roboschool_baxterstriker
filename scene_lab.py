import os
import numpy as np
from roboschool.scene_abstract import Scene, cpp_household

class LabScene(Scene):
    # zero_at_running_strip_start_line = True   # if False, center of coordinates (0,0,0) will be at the middle of the stadium
    # stadium_halflen   = 105*0.25    # FOOBALL_FIELD_HALFLEN
    # stadium_halfwidth = 50*0.25     # FOOBALL_FIELD_HALFWID

    def episode_restart(self):
        Scene.episode_restart(self)   # contains cpp_world.clean_everything()
        lab_ground_pose = cpp_household.Pose()
        lab_ground_pose.set_xyz(0, 0, 1)
        lab_ground_pose.set_rpy(np.pi/2, 0, 0)

        # if self.zero_at_running_strip_start_line:
        # lab_pose.set_xyz(0, 0, 0)  # see RUN_STARTLINE, RUN_RAD constants
        # scale seems not working
        self.lab_ground = self.cpp_world.load_thingy(
            os.path.join(os.path.dirname(__file__), "models_indoor/floor.obj"),
            lab_ground_pose, 1.0, 0, 0xFFFFFF, True)

        table_pose = cpp_household.Pose()
        table_pose.set_rpy(0, 0, np.pi/2)
        table_pose.set_xyz(0.7, 0, 0.75)
        self.table = self.cpp_world.load_urdf(
            os.path.join(os.path.dirname(__file__), "models_indoor/lab/bordered_table.urdf"),
            table_pose, False, False)

        slope_pose = cpp_household.Pose()
        slope_pose.set_rpy(0, 0, np.pi/2)
        slope_pose.set_xyz(0.6, -0.4, 0.8)
        self.slope = self.cpp_world.load_urdf(
            os.path.join(os.path.dirname(__file__), "models_indoor/lab/slope.urdf"),
            slope_pose, False, False)


        ball_pose = cpp_household.Pose()
        ball_pose.set_xyz(0.6, -0.4, 0.88)
        self.ball = self.cpp_world.load_urdf(
            os.path.join(os.path.dirname(__file__), "models_indoor/lab/ball.urdf"),
            ball_pose, False, False
        )
        self.ball_home_pose = ball_pose
        self.ball_r = 0.025

        frame_pose = cpp_household.Pose()
        frame_pose.set_xyz(.95, 0.4, 0.88)
        frame_pose.set_rpy(0, 0, np.pi/2)

        self.frame = self.cpp_world.load_urdf(
            os.path.join(os.path.dirname(__file__), "models_indoor/lab/frame.urdf"),
            frame_pose, False, False
        )
        self.frame_home_pose = frame_pose
        self.frame_width = 0.14
        self.frame_height = 0.07

        self.ground_plane_mjcf = self.cpp_world.load_mjcf(os.path.join(os.path.dirname(__file__), "mujoco_assets/ground_plane.xml"))
    


class SinglePlayerLabScene(LabScene):
    "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
    multiplayer = False

class MultiplayerLabScene(LabScene):
    multiplayer = True
    players_count = 3
    def actor_introduce(self, robot):
        LabScene.actor_introduce(self, robot)
        i = robot.player_n - 1  # 0 1 2 => -1 0 +1
        robot.move_robot(0, i, 0)
