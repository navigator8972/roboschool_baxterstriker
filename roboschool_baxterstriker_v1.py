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
    def __init__(self, width=320, height=200, caption='test_window'):
        pw.Window.__init__(self, width=width, height=height, vsync=False, resizable=True, caption=caption)
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
        self.clear = 0

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
        if self.keys.get(ord('C'), 0) > 0:
            self.clear = 1

'''
gmm control with vaedyn model and image input
'''
'''
prepare tf model to encode and predict image input
'''
import tensorflow as tf

import sys
import os
import cPickle as cp
from il_models.model_vaedyn import VAEDYN
import numpy as np

sample_mode = True

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'il_models', 'save-vaedyn', 'config.pkl')) as f:
    saved_args = cp.load(f)

model = VAEDYN(saved_args, sample_mode)
saver = tf.train.Saver(tf.global_variables())
# saver = tf.train.Saver()
sess = tf.InteractiveSession()



#if no model folder is specified
ckpt = tf.train.get_checkpoint_state(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'il_models', 'save-vaedyn'))
model_checkpoint_path = ckpt.model_checkpoint_path

print "loading model: ",model_checkpoint_path

saver.restore(sess, model_checkpoint_path)

#auxiliary function to update the encode
def update_encode(curr_state, input_img=None):
    #the previous encode is 8-dim, with 4 as c, 4 as h
    #return img_encode, (z, h) and next (c, h)
    state_c, state_h = curr_state[:, :4], curr_state[:, 4:]
    if input_img is None:
        #use prior model to predict next encode
        feed = {model.initial_state_c:state_c,
                model.initial_state_h:state_h}
        prior_mu, prior_sigma = sess.run([model.prior_mu, model.prior_sigma], feed)
        z = prior_mu[0]+np.random.randn()*prior_sigma[0]*0
        next_state_c, next_state_h = sess.run([model.final_state_c, model.final_state_h], {model.cell.z:z, model.initial_state_c:state_c, model.initial_state_h:state_h})
        img_encode = np.concatenate([prior_mu[0], prior_sigma[0], state_h], axis=1)
    else:
        #we have observation now, update the belief of latent code
        feed = {model.initial_state_c:state_c,
                model.initial_state_h:state_h,
                model.input_data: input_img}
        enc_mu, enc_sigma = sess.run([model.enc_mu_post, model.enc_sigma_post], feed)
        z = enc_mu[0]+np.random.randn()*enc_sigma[0]*0
        next_state_c, next_state_h = sess.run([model.final_state_c, model.final_state_h], {model.cell.z:z, model.initial_state_c:state_c, model.initial_state_h:state_h})
        img_encode = np.concatenate([enc_mu[0], enc_sigma[0], state_h], axis=1)
    next_state = np.concatenate([next_state_c, next_state_h], axis=1)

    return img_encode[0], next_state

from PIL import Image

def thumbnail_img(img):
    img_from_array = Image.fromarray(img)
    img_from_array.thumbnail((img_from_array.size[0]/2, img_from_array.size[1]/2), Image.ANTIALIAS)
    return np.asarray(img_from_array)

def flatten_helper(d):
    return d.flatten().astype(np.float32)*1./255

#prepare seed to have stable shuffle
# np.random.seed(1234)  # not for testing environment
tf.set_random_seed(1234)

import gmr.gmr.gmm as gmm
gmm_vaeenc_bp_model_file = 'il_models/gmm_vaeenc_bp.gmm'
gmm_vaeenc_sp_model_file = 'il_models/gmm_vaeenc_sp.gmm'
class TestGMMVAEDynControl:
    def __init__(self):
        #the n_components is temporary, the loaded model will overwrite it
        self.gmm_vaeenc_bp = gmm.GMM(n_components=2)
        self.gmm_vaeenc_bp.load_model(gmm_vaeenc_bp_model_file)
        self.gmm_vaeenc_sp = gmm.GMM(n_components=2)
        self.gmm_vaeenc_sp.load_model(gmm_vaeenc_sp_model_file)
        return

    def predict_paddler_pos(self, enc):
        paddler_x_pos = self.gmm_vaeenc_bp.predict(range(12), np.array([enc]))
        return paddler_x_pos[0]+0.582752705075  #remember to compensate the mean

    def predict_paddler_action(self, enc):
        #evaluate the likelihood, simply use a threshold to decide if we should act
        density = self.gmm_vaeenc_sp.to_probability_density(np.array([enc]))
        
        return density[0] > 1e-3


import time
import cPickle as cp

def main(args):
    env = gym.make("RoboschoolBaxterStriker-v0")

    wrist_cam_window = PygletWindow(100, 100, 'Camera')
    # main_window = PygletWindow(1280, 1024)

    ctrl = TestKeyboardControl()
    cb_set = False
    reset_called = False

    record_data = True

    '''
    prepare the gmm control with real position accessed
    '''
    gmm_vaedyn_ctrl = TestGMMVAEDynControl()

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
                # print ball_pose.xyz()
                rand_vel = np.random.rand()*0.16 - 0.08
                print('Random perturbation: ', rand_vel)
                env.unwrapped.scene.ball.set_pose_and_speed(ball_pose, rand_vel, 0.2, 0)
                # env.unwrapped.scene.ball.set_pose_and_speed(ball_pose, -0.05, 0.2, 0)
                update_frame = True
                
                #initialize cdssmoother with the current x position
                paddle_pose = env.unwrapped.get_current_paddle_pose()
                smoother = CDSSmoother(dt=0.01)
                smoother.set_omega(10)
                smoother.set_state(np.array([paddle_pose[0]]))
                smoother.set_target(np.array([paddle_pose[0]]))

                #initialize state
                curr_state = np.zeros((1, saved_args.rnn_size*2))
            
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

            env.unwrapped.wrist_camera_adjust()
            rgb, _, _, _, _ = env.unwrapped.wrist_camera.render(False, False, False)
            # env.unwrapped.wrist_camera.test_window()
            rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (100,100,3) )
            if frame > 50:
                #show nothing as we will not have camera input anymore
                rgb[:] = 0
            wrist_cam_window.imshow(rgb)

            '''
            section for gmm model-based control
            use the rgb for the first few steps, and then model based
            '''
            strike_or_not = False
            if update_frame:
                #prepare the encode
                if frame % 10 == 0:
                    if frame < 51:
                        #we can use image here
                        # rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (100,100,3) )
                        input_img = np.zeros((1, 1, 7500))
                        input_img[0][0] = flatten_helper(thumbnail_img(rgb))

                    else:
                        #we dont have sensor input now, we have to rely on the model prediction
                        input_img = None
                    enc, next_state = update_encode(curr_state, input_img)
                    curr_state = next_state
                    #use gmm on the encoded information
                    # print 'enc:', enc
                    paddler_x_pos = gmm_vaedyn_ctrl.predict_paddler_pos(enc[:12])
                    strike_or_not = gmm_vaedyn_ctrl.predict_paddler_action(enc[8:]) 
                    #exercise these motions
                    if frame < 71:
                        #set the target
                        smoother.set_target(np.array([paddler_x_pos[0]]))
                        # print smoother.update()[0]
                smoothed_cmd = smoother.update()[0]
                # if frame % 10 == 0:
                #     print frame, paddler_x_pos[0], strike_or_not, smoothed_cmd
                if frame % 1 == 0 and frame < 71:
                    env.unwrapped.set_paddle_xpos(smoothed_cmd)
                if strike_or_not:
                    a[-1] = 1
                
            obs, r, done, _ = env.step(a)

            score += r

            if ctrl.clear == 1:
                score = 0
                ctrl.clear = 0

            still_open = env.render("human")
            env.unwrapped.scene.cpp_world.test_window_score('%04i %.4f' % (frame, score))

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
