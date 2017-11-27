import gym, roboschool
import numpy as np
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from scene_lab import SinglePlayerLabScene
from roboschool.scene_abstract import cpp_household
from roboschool.scene_stadium import SinglePlayerStadiumScene
import os, sys

#PyKDL needs python 2.7
import baxter_pykdl

class RoboschoolBaxterStriker(RoboschoolUrdfEnv):
    def __init__(self):
        RoboschoolUrdfEnv.__init__(self,    "baxter_description/urdf/baxter_striker.urdf",
                                            'left_arm',
                                            action_dim=7,
                                            obs_dim=6+7,
                                            fixed_base=True,
                                            self_collision=False)   #do not use self collision like Atlas
        
        self.jnt_name = [   'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2',
                            'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']

        self.jnt_name_to_ind = {'left_s0':0, 'left_s1':1, 'left_e0':2, 'left_e1':3, 'left_w0':4, 'left_w1':5, 'left_w2':6,
                                'right_s0':7, 'right_s1':8, 'right_e0':9, 'right_e1':10, 'right_w0':11, 'right_w1':12, 'right_w2':13 }
        self.untuck_pos = [-0.08, -1.0, -1.19, 1.94,  0.67, 1.03, -0.50, 0.08, -1.0,  1.19, 1.94, -0.67, 1.03,  0.50]
        self.ready_pos = [-0.08, -1.0, -1.19, 1.94,  0.67, 1.03, -0.50, 0.08, -1.0,  1.19, 1.94, -0.67, 1.03,  0.50]
        self.prepare_baxter_robot_manipulator()

        self.strike_counter = 0
        self.strike_timeout = 15
        self.strike_activated = False
        self.max_strike_pos = np.pi/6

        self.bullet_queried = False
        self.ball_has_passed_the_frame = False
        return

    def prepare_baxter_robot_manipulator(self):
        path_prefix = os.path.dirname(os.path.abspath(__file__))
        #urdf
        self.baxter_urdf_path = os.path.join(path_prefix, 'models_robot/baxter_description/urdf/baxter.urdf')

        # use revised baxter_pykdl to create inverse kinemtics model
        self.robot_dynamics_left_arm = baxter_pykdl.baxter_pykdl.baxter_dynamics('left', self.baxter_urdf_path)
        self.robot_dynamics_right_arm = baxter_pykdl.baxter_pykdl.baxter_dynamics('right', self.baxter_urdf_path)
        #print structure
        print('Baxter Left Arm Description')
        self.robot_dynamics_left_arm.print_robot_description()
        print('Baxter Right Arm Description')
        self.robot_dynamics_right_arm.print_robot_description()
        return

    def create_single_player_scene(self):
        # return SingleRobotEmptyScene(gravity=9.8, timestep=0.0165/8, frame_skip=8)   # 8 instead of 4 here  #so this seems to be related to the control frequency...
        # return SingleRobotEmptyScene(gravity=9.8, timestep=0.0165, frame_skip=1)
        return SinglePlayerLabScene(gravity=9.8, timestep=0.0165/8, frame_skip=8)   # 8 instead of 4 here  #so this seems to be related to the control frequency...

    def calc_state(self):
        s = np.zeros(self.observation_space.shape)
        if self.bullet_queried:
            #the state we need: ball position, ball velocity, endeffector pose
            s[:3] = self.scene.ball.root_part.pose().xyz()
            s[3:6] = self.scene.ball.root_part.speed()
            s[6:] = self.get_current_paddle_pose()
        
        return s

    def robot_specific_reset(self):
        self.bullet_queried = False
        self.set_initial_pose()
        #after rest, the position will be zero if this is not set. Setting this makes the value non-zero but the actual pose is not changed...
        # self.set_untuck_position()  
        self.set_ready_position()
        self.right_camera_link = self.parts['right_hand_camera']
        self.wrist_camera = self.scene.cpp_world.new_camera_free_float(100, 100, "wrist_camera")
        self.wrist_camera_adjust()
        return
    
    def set_ready_position(self):
        ready_pos_bkp = [p for p in self.ready_pos]
        ready_angles = self.ready_pos[:7]
        ready_pose = self.robot_dynamics_left_arm.forward_position_kinematics(ready_angles)
        ready_pose_shift = ready_pose[:]
        ready_pose_shift[2]-=0.35   #z axis
        ready_pose_shift[1]+=0.2    #y axis
        ready_angles_shift = self.robot_dynamics_left_arm.inverse_kinematics(ready_pose_shift[:3], ready_pose_shift[3:], ready_angles)

        self.ready_pos[:7] = ready_angles_shift
        self.updated_left_arm_pos = ready_angles_shift

        #for camera
        camera_angles = self.ready_pos[7:]
        camera_pose = self.robot_dynamics_right_arm.forward_position_kinematics(camera_angles)
        camera_pose_shift = camera_pose[:]
        camera_pose_shift[2]+= 0.05
        camera_pose_shift[1]+= 0.1
        camera_pose_shift[0]+= 0.2
        camera_angles_shift = self.robot_dynamics_right_arm.inverse_kinematics(camera_pose_shift[:3], camera_pose_shift[3:], camera_angles)
        self.ready_pos[7:] = camera_angles_shift
        for name, joint in self.jdict.items():
            if name in self.jnt_name_to_ind:
                # print 'assigning joint {0} to position {1}.'.format(name, self.untuck_pos[self.jnt_name_to_ind[name]])
                joint.reset_current_position(self.ready_pos[self.jnt_name_to_ind[name]], 0)
                joint.set_servo_target(self.ready_pos[self.jnt_name_to_ind[name]], 1, .1, 100)
        
        # self.set_joint_torques(left=True, trq=np.zeros(7))
        self.ready_pos_shift = self.ready_pos
        self.ready_pos = ready_pos_bkp
        self.ready_pose_shift = ready_pose_shift
        return

    def set_untuck_position(self):
        for name, joint in self.jdict.items():
            if name in self.jnt_name_to_ind:
                # print 'assigning joint {0} to position {1}.'.format(name, self.untuck_pos[self.jnt_name_to_ind[name]])
                joint.reset_current_position(self.untuck_pos[self.jnt_name_to_ind[name]], 0)
                joint.set_servo_target(self.untuck_pos[self.jnt_name_to_ind[name]], 1, .1, 100)
        
        # self.set_joint_torques(left=True, trq=np.zeros(7))
        return

    def set_joint_torques(self, left, trq):
        assert len(trq) == 7
        if left:
            offset = 0
        else:
            offset = 7
        
        for ind, t in enumerate(trq):
            self.jdict[self.jnt_name[ind+offset]].set_motor_torque(t)

        return

    def set_paddle_xpos(self, xpos):
        curr_jnt_pos = self.get_current_arm_state(left=True)[:, 0]

        curr_pose = self.robot_dynamics_left_arm.forward_position_kinematics(curr_jnt_pos)
        curr_pose[0] = xpos
        curr_pose[0] = max(curr_pose[0], self.ready_pose_shift[0]-0.05)  #truncate +-10cm
        curr_pose[0] = min(curr_pose[0], self.ready_pose_shift[0]+0.15)
        curr_pose[1] = self.ready_pose_shift[1]
        curr_pose[2] = self.ready_pose_shift[2]
        
        update_jnt_pos = self.robot_dynamics_left_arm.inverse_kinematics(curr_pose[:3], self.ready_pose_shift[3:], curr_jnt_pos)
        if update_jnt_pos is not None:
            for i in range(7):
                self.jdict[self.jnt_name[i]].set_servo_target(update_jnt_pos[i], 1, .1, 100)
        self.updated_left_arm_pos = update_jnt_pos
        return
    
    def move_paddle(self, forward=True):
        #move the paddle forward or backward
        stride = 0.003
        curr_jnt_pos = self.get_current_arm_state(left=True)[:, 0]

        curr_pose = self.robot_dynamics_left_arm.forward_position_kinematics(curr_jnt_pos)
        curr_pose[0] += stride * (1 if forward else -1)
        curr_pose[0] = max(curr_pose[0], self.ready_pose_shift[0]-0.05)  #truncate +-10cm
        curr_pose[0] = min(curr_pose[0], self.ready_pose_shift[0]+0.15)
        curr_pose[1] = self.ready_pose_shift[1]
        curr_pose[2] = self.ready_pose_shift[2]

        update_jnt_pos = self.robot_dynamics_left_arm.inverse_kinematics(curr_pose[:3], self.ready_pose_shift[3:], curr_jnt_pos)
        if update_jnt_pos is not None:
            for i in range(7):
                self.jdict[self.jnt_name[i]].set_servo_target(update_jnt_pos[i], 1, .1, 100)
        self.updated_left_arm_pos = update_jnt_pos
        return

    def strike_motion(self):
        if self.strike_activated:
            if self.strike_counter > self.strike_timeout:
                self.strik_counter = 0
                self.strike_activated = False
                self.jdict['left_w2'].set_servo_target(self.updated_left_arm_pos[self.jnt_name_to_ind['left_w2']], 1, .1, 100)
            else:
                #calculate
                ratio = self.strike_counter/float(self.strike_timeout)
                
                if ratio < 0.5:
                    wrist_twist_pos = self.max_strike_pos * 4 * ratio**2
                else:
                    wrist_twist_pos = self.max_strike_pos * 4 * max(1-ratio, 0.0)**2
                # print('Set motor position.')
                self.jdict['left_w2'].set_servo_target(self.updated_left_arm_pos[self.jnt_name_to_ind['left_w2']]-wrist_twist_pos, 1, .1, 100)
                self.strike_counter+=1
        return

    def strike(self):
        if not self.strike_activated:
            self.strike_counter = 0
            self.strike_activated = True
        return
        
    def make_joint_angles(self, angles, left=True):
        assert len(angles) == 7
        if left:
            offset = 0
        else:
            offset = 7
        res_dict = dict()
        for ind, theta in enumerate(angles):
            res_dict[self.jnt_name[ind+offset]] = theta
        return res_dict

    def isOnTheGoal(self):
        if not self.ball_has_passed_the_frame:
            ball_pos = self.scene.ball.root_part.pose().xyz()
            frame_pos = self.scene.frame_home_pose.xyz()
            if ball_pos[0] >= frame_pos[0]-0.035 and ball_pos[0] <= frame_pos[0]+0.035  \
                and ball_pos[1] >= frame_pos[1]-self.scene.frame_width/2+self.scene.ball_r and ball_pos[1] <= frame_pos[1]+self.scene.frame_width/2-self.scene.ball_r   \
                and ball_pos[2] >= frame_pos[2]-self.scene.frame_height-0.05+self.scene.ball_r/2 and ball_pos[2] <= frame_pos[2]-0.05-self.scene.ball_r:
                # print 'Goal!'
                self.ball_has_passed_the_frame = True
                return 1
        return 0

    def _step(self, action):
        self.apply_action(action)
        self.scene.global_step()

        state = self.calc_state()
        done = False
        self.rewards = [self.isOnTheGoal()]
        self.HUD(state, action, done)

        if not self.bullet_queried:
            self.bullet_queried = True
        return state, sum(self.rewards), done, {}

    def apply_action(self, a):
        if a[-1] > 0:
            # print("Initiate a strike!")
            self.strike()
        if a[-2] == 1:
            self.move_paddle(forward=True)
        elif a[-2] == -1:
            self.move_paddle(forward=False)
        self.strike_motion()
        return

    def camera_adjust(self):
        # self.camera.move_and_look_at(0,1.2,1.2, 0,0,0.5)
        self.camera.move_and_look_at(1.5, 1.5, 1.5, 0, 0, 0)
        return

    def human_camera_adjust(self):
        self.human_camera.move_and_look_at(1.5, 1.5, 1.5, 0, 0, 0)
        return

    def wrist_camera_adjust(self):
        loc = self.right_camera_link.pose().xyz()
        rel_pose = cpp_household.Pose()
        rel_pose.set_xyz(0, 0, 1)   #look towards +z direction in the camera frame
        disp = self.right_camera_link.pose().dot(rel_pose)
        tar = disp.xyz() + loc
        self.wrist_camera.move_and_look_at(loc[0], loc[1], loc[2], tar[0], tar[1], tar[2])
        # print loc, tar
        # self.wrist_camera.move_and_look_at(1,1,1,0,0,0)
        return

    def set_initial_pose(self):
        # self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(0, 0, 0.925)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)
    
    def reset_ball_position(self):
        self.scene.ball.set_pose_and_speed(self.scene.ball_home_pose, 0, 0, 0)
        self.ball_has_passed_the_frame = False
        return
    
    def get_current_paddle_pose(self):
        arm_jnt = self.get_current_arm_state(left=True)[:, 0]
        curr_pose = self.robot_dynamics_left_arm.forward_position_kinematics(arm_jnt)
        return curr_pose

    def get_current_arm_state(self, left=True):
        # self.cpp_robot.query_position()
        if left:
            offset = 0
        else:
            offset = 7
        
        res = np.array([self.jdict[self.jnt_name[i+offset]].current_position() for i in range(7)])
        return res

