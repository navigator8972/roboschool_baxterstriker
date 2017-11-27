import os.path, time, gym
import argparse

import numpy as np

import pyglet, pyglet.window as pw, pyglet.window.key as pwk
from pyglet import gl

#workaround to resolve OpenGL when nVidia driver is used, see:
#https://bugs.launchpad.net/ubuntu/+source/python-qt4/+bug/941826
# from OpenGL import GL
from OpenGL import GLU

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
import roboschool

#register here
from gym.envs.registration import register
register(
    id='RoboschoolBaxterStriker-v0',
    entry_point='roboschool_baxterstriker:RoboschoolBaxterStriker',
    max_episode_steps=1000
)

from cds_smoother import CDSSmoother

class PygletWindow(pw.Window):
    def __init__(self, width=320, height=200):
        pw.Window.__init__(self, width=width, height=height, vsync=False, resizable=True)
        self.still_open = True

        @self.event
        def on_close():
            self.still_open = False

        @self.event
        def on_resize(width, height):
            self.win_w = width
            self.win_h = height

    def imshow(self, arr):
        H, W, C = arr.shape
        assert C==3
        image = pyglet.image.ImageData(W, H, 'RGB', arr.tobytes(), pitch=W*-3)
        self.clear()
        self.switch_to()
        self.dispatch_events()
        texture = image.get_texture()
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture.width  = W
        texture.height = H
        texture.blit(0, 0, width=self.win_w, height=self.win_h)
        self.flip()

    def each_frame(self):
        return


class TestKeyboardControl:
    def __init__(self):
        self.keys = {}
        self.strike = 0
        self.reset = 0
        self.start = 0
        self.untuck = 0
        self.move = 0
        self.home = 0
        self.save = 0

    def key(self, event_type, key, modifiers):
        self.keys[key] = +1 if event_type==6 else 0
        # print ("event_type", event_type, "key", key, "modifiers", modifiers)
        # case insensitive
        if self.keys.get(ord('A'), 0) > 0:
            self.strike = 1
        if self.keys.get(ord('S'), 0) > 0:
            self.start = 1
        if self.keys.get(ord('R'), 0) > 0:
            self.reset = 1
        if self.keys.get(ord('U'), 0) > 0:
            self.untuck = 1
        if self.keys.get(ord('Q'), 0) > 0:
            self.move = 1
        if self.keys.get(ord('E'), 0) > 0:
            self.move = -1
        if self.keys.get(ord('H'), 0) > 0:
            self.home = 1
        if self.keys.get(ord('D'), 0) > 0:
            self.save = 1

'''
gmm control with real position accessed
'''
import gmr.gmr.gmm as gmm
gmm_realpos_bp_model_file = 'il_models/gmm_realpos_bp.gmm'
gmm_realpos_sp_model_file = 'il_models/gmm_realpos_sp.gmm'
class TestGMMRealPosControl:
    def __init__(self):
        #the n_components is temporary, the loaded model will overwrite it
        self.gmm_realpos_bp = gmm.GMM(n_components=2)
        self.gmm_realpos_bp.load_model(gmm_realpos_bp_model_file)
        self.gmm_realpos_sp = gmm.GMM(n_components=2)
        self.gmm_realpos_sp.load_model(gmm_realpos_sp_model_file)
        return

    def predict_paddler_pos(self, ball_pos):
        paddler_x_pos = self.gmm_realpos_bp.predict(range(3), np.array([ball_pos]))
        return paddler_x_pos[0]

    def predict_paddler_action(self, ball_pos_xy):
        #evaluate the likelihood, simply use a threshold to decide if we should act
        density = self.gmm_realpos_sp.to_probability_density(np.array([ball_pos_xy]))
        return density[0] > 0.01


import time
import cPickle as cp

def main(args):
    env = gym.make("RoboschoolBaxterStriker-v0")

    # wrist_cam_window = PygletWindow()
    # main_window = PygletWindow(1280, 1024)

    ctrl = TestKeyboardControl()
    cb_set = False
    reset_called = False

    record_data = True

    '''
    prepare the gmm control with real position accessed
    '''
    gmm_real_ctrl = TestGMMRealPosControl()



    max_frame = 150
    still_open = True
    while still_open:
        frame = 0
        score = 0
        restart_delay = 0.0
        obs = env.reset()

        reset_cnt = 10

        if not cb_set:
            env.unwrapped.scene.cpp_world.set_key_callback(ctrl.key)
            cb_set = True

        a = np.zeros(env.action_space.shape)
        
        data = {'images':[None for _ in range(max_frame)], 'states':[None for _ in range(max_frame)], 'actions':[None for _ in range(max_frame)], 'rewards':[None for _ in range(max_frame)]}
        update_frame = False

        while still_open:
            if ctrl.start == 1:
                print('Ball starts rolling! Recording...')
                ctrl.start = 0
                #initiate a velocity to the ball
                ball_pose = env.unwrapped.scene.ball.root_part.pose()
                env.unwrapped.scene.ball.set_pose_and_speed(ball_pose, np.random.rand()*0.16-0.08, 0.2, 0)
                # env.unwrapped.scene.ball.set_pose_and_speed(ball_pose, -0.08, 0.2, 0)
                update_frame = True
                
                #initialize cdssmoother with the current x position
                paddle_pose = env.unwrapped.get_current_paddle_pose()
                smoother = CDSSmoother(dt=0.01)
                smoother.set_omega(10)
                smoother.set_state(np.array([paddle_pose[0]]))
                smoother.set_target(np.array([paddle_pose[0]]))
            
            if ctrl.home == 1:
                #this is a light-weight reset: only move the ball to the initial position
                #but not reset the score
                ctrl.home = 0
                env.unwrapped.reset_ball_position()
                data = {'images':[None for _ in range(max_frame)], 'states':[None for _ in range(max_frame)], 'actions':[None for _ in range(max_frame)], 'rewards':[None for _ in range(max_frame)]}
                update_frame = False
                frame = 0
            a = np.zeros(env.action_space.shape)
            if ctrl.strike == 1:
                ctrl.strike = 0
                #initiate the stroke motion of the robot
                a[-1] = 1
            else:
                a[-1] = 0

            if ctrl.move != 0:
                a[-2] = ctrl.move
                ctrl.move = 0
            else:
                a[-2] = 0


            '''
            section for gmm real pos control
            use the obs
            '''
            strike_or_not = False
            if update_frame:
                if frame % 1 == 0:
                    paddler_x_pos = gmm_real_ctrl.predict_paddler_pos(obs[:3])
                    strike_or_not = gmm_real_ctrl.predict_paddler_action(obs[:2])                
                #exercise these motions
                if frame % 10 == 0 and frame < 71:
                    #set the target
                    smoother.set_target(np.array([paddler_x_pos[0]]))
                    # print smoother.update()[0]
                smoothed_cmd = smoother.update()[0]
                # print(frame, paddler_x_pos, strike_or_not, smoothed_cmd)
                if frame % 1 == 0 and frame < 71:
                    env.unwrapped.set_paddle_xpos(smoothed_cmd)
                if strike_or_not:
                    a[-1] = 1
                
            obs, r, done, _ = env.step(a)
            # print obs
            #it seems that the bullet joint position is set to the correct value and the rendering sucks
            #but if i check the wrist_camera render it apparently shows the incorrect image
            #there seems to be some inconsistency between the wrapped value and real bullet state
            # print('Left:', env.unwrapped.get_current_arm_state(left=True)[:, 0])
            # print('Right:', env.unwrapped.get_current_arm_state(left=False)[:, 0])

            # if reset_cnt > 0:
            #     env.unwrapped.robot_specific_reset()
            #     reset_cnt-=1

            score += r

            still_open = env.render("human")
            env.unwrapped.scene.cpp_world.test_window_score('%04i %.4f' % (frame, score))
            # <hyin/Aug-31st-2017> so it seems env.render impacts the stability of reset
            # running env.step and call an env.reset is okay
            # but the issue appears if env.render("human") is called
            # same for wrist_camera. so Oleg's fix in render seems not complete for me.
            # if not reset_called:
            #     reset_called = True
            #     break

            # if not reset_called:
            #     reset_called = True
            #     break

            # env.unwrapped.wrist_camera_adjust()
            # rgb, _, _, _, _ = env.unwrapped.wrist_camera.render(False, False, False)
            # env.unwrapped.wrist_camera.test_window()

            # section for keyboard control
            if ctrl.untuck == 1:
                #set untuck
                env.unwrapped.set_untuck_position()
                ctrl.untuck = 0

            if ctrl.reset == 1:
                #reset
                ctrl.reset = 0
                ctrl.start = 0
                break

            if ctrl.save == 1:
                #save current data
                if data['images'][-1] is not None:
                    #valid data
                    fname = time.strftime("%Y%m%d%H%M%S")
                    fpath = os.path.join('bin', fname+'.pkl')
                    cp.dump(data, open(fpath, 'wb'))
                    print('Data has been dumped into {0}'.format(fpath))
                else:
                    print('Recorded data is not complete. Skip dumping.')
                ctrl.save = 0
            
            if update_frame:
                if frame >= 150:
                    print('Finishing this episode. Press D to dump the data.')
                    frame = 0
                    update_frame = False
                else:
                    # print(frame)
                    # data['images'][frame] = rgb
                    # data['states'][frame] = obs
                    # data['actions'][frame] = a[:]
                    # data['rewards'][frame] = r
                    frame += 1



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dummy', type=int, default=3,
                        help='dummy argument')
    args = parser.parse_args()
    main(args)
