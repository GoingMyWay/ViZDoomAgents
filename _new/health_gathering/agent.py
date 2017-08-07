# coding: utf-8
# implement of agent
import time
import random
import pickle
import matplotlib
gui_env = [i for i in matplotlib.rcsetup.interactive_bk]
for gui in gui_env:
    print("testing", gui)
    try:
        matplotlib.use(gui, warn=False, force=True)
        from matplotlib import pyplot as plt
        print("Using ..... ", matplotlib.get_backend())
    except:
        print("    ", gui, "Not found")
import itertools as it

import numpy as np
import tensorflow as tf
from vizdoom import *

import utils
from . import configs as cfg
from . import network


class Agent(object):
    """
    Agent
    """
    def __init__(self, sess, game, name, optimizer=None, global_episodes=None, play=False):
        self.summary_step = 3

        self.name = "worker_" + str(name)
        self.number = name

        self.episode_reward = []
        self.episode_episode_total_pickes = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.episode_health = []

        # Create the local copy of the network and the tensorflow op to
        # copy global parameters to local network
        if not play:
            self.trainer = optimizer
            self.global_episodes = global_episodes
            self.increment = self.global_episodes.assign_add(1)
            self.local_AC_network = network.ACNetwork(self.name, optimizer, play=play)
            self.summary_writer = tf.summary.FileWriter("./summaries/healthpack/ag_%s" % str(self.number), sess.graph)
            self.update_local_ops = tf.group(*utils.update_target_graph('global', self.name))
        else:
            self.local_AC_network = network.ACNetwork(self.name, optimizer, play=play)

        if not isinstance(game, Game):
            raise TypeError("Type Error")
        game.init_game()
        if play:
            game.add_game_args("+viz_render_all 1")
            game.set_render_hud(False)
            game.set_ticrate(35)
        game.init()
        self.env = game
        self.actions = game.get_actions()

    def get_feed_dict(self, roll_out, bootstrap_value):
        roll_out = np.array(roll_out)
        observations = roll_out[:, 0]
        actions = roll_out[:, 1]
        rewards = roll_out[:, 2]
        next_observations = roll_out[:, 3]
        values = roll_out[:, 5]

        # The advantage function uses "Generalized Advantage Estimation"
        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = utils.discount(rewards_plus, cfg.gamma)[:-1]
        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + cfg.gamma * value_plus[1:] - value_plus[:-1]
        advantages = utils.discount(advantages, cfg.gamma)

        feed_dict = {
            self.local_AC_network.target_v: discounted_rewards,
            self.local_AC_network.inputs: np.stack(observations),
            self.local_AC_network.actions: actions,
            self.local_AC_network.advantages: advantages
        }
        return feed_dict

    def infer(self, sess, feed_dict, roll_out):
        l, v_l, p_l, e_l, g_n, v_n = sess.run([
            self.local_AC_network.loss,
            self.local_AC_network.value_loss,
            self.local_AC_network.policy_loss,
            self.local_AC_network.entropy,
            self.local_AC_network.grad_norms,
            self.local_AC_network.var_norms],
            feed_dict=feed_dict)
        return l / len(roll_out), v_l / len(roll_out), p_l / len(roll_out), e_l / len(roll_out), g_n, v_n

    def train_a3c(self, sess, coord, saver):
        if not isinstance(saver, tf.train.Saver):
            raise TypeError('saver should be tf.train.Saver')

        episode_count = sess.run(self.global_episodes)
        start_t = time.time()
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                self.local_AC_network.pull(session=sess)  # update local ops in every episode

                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                last_total_picked_packs = 0

                self.env.new_episode()
                episode_st = time.time()
                while not self.env.is_episode_finished():

                    # if utils.check_play(self.env.get_state()):
                    s = self.env.get_state().screen_buffer
                    s = utils.process_frame(s, cfg.img_dim)
                    # Take an action using probabilities from policy network output.
                    a_index, v = self.local_AC_network.get_action_index_and_value(
                                    sess, {self.local_AC_network.inputs: [s]})

                    # make an action
                    self.env.make_action(self.actions[a_index], 4)

                    picked_pack_num = doom_fixed_to_double(self.env.get_game_variable(GameVariable.USER1)) / 100.
                    picked_delta = int(picked_pack_num - last_total_picked_packs)
                    last_total_picked_packs += picked_delta

                    reward = self.reward_function(picked_delta, episode_step_count)
                    episode_reward += reward

                    d = self.env.is_episode_finished()
                    if d:
                        s1 = s
                    else:  # game is not finished
                        s1 = self.env.get_state().screen_buffer
                        s1 = utils.process_frame(s1, cfg.img_dim)

                    episode_buffer.append([s, a_index, reward, s1, d, v])
                    episode_values.append(v)
                    # summaries information
                    s = s1
                    episode_step_count += 1

                    if len(episode_buffer) == 32 and d is False and episode_step_count != cfg.max_episode_length - 1:
                        v1 = self.local_AC_network.get_value(sess, {self.local_AC_network.inputs: [s]})
                        feed_dict = self.get_feed_dict(episode_buffer, v1)

                        l, v_l, p_l, e_l, g_n, v_n = self.infer(sess, feed_dict, episode_buffer)
                        self.local_AC_network.push(sess, feed_dict)  # update the global network
                        self.local_AC_network.pull(sess)  # update local network

                        episode_buffer = []
                    if d is True:
                        self.episode_health.append(self.env.get_game_variable(GameVariable.HEALTH))
                        print('{}, picks: {}, episode #{}, reward: {}, steps:{}, time costs:{}'.format(
                            self.name, last_total_picked_packs, episode_count,
                            episode_reward, episode_step_count, time.time()-episode_st))
                        break

                # summaries
                self.episode_reward.append(episode_reward)
                self.episode_episode_total_pickes.append(last_total_picked_packs)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    feed_dict = self.get_feed_dict(episode_buffer, 0.0)
                    l, v_l, p_l, e_l, g_n, v_n = self.infer(sess, feed_dict, episode_buffer)
                    self.local_AC_network.push(sess, feed_dict)
                    self.local_AC_network.pull(sess)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 50 == 0 and self.name == 'worker_0':
                        saver.save(sess, cfg.MODEL_PATH+'/model-'+str(episode_count)+'.ckpt')
                        print("Episode count {}, saved Model, time costs {}".format(episode_count, time.time()-start_t))
                        start_t = time.time()

                    mean_picked = np.mean(self.episode_episode_total_pickes[-5:])
                    mean_reward = np.mean(self.episode_reward[-5:])
                    mean_health = np.mean(self.episode_health[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Performance/Total Picks', simple_value=mean_picked)
                    summary.value.add(tag='Performance/Reward', simple_value=mean_reward)
                    summary.value.add(tag='Performance/Health', simple_value=mean_health)
                    summary.value.add(tag='Performance/Steps', simple_value=mean_length)
                    summary.value.add(tag='Performance/Value', simple_value=mean_value)
                    summary.value.add(tag='Losses/Total Loss', simple_value=l)
                    summary.value.add(tag='Losses/Value Loss', simple_value=v_l)
                    summary.value.add(tag='Losses/Policy Loss', simple_value=p_l)
                    summary.value.add(tag='Losses/Entropy', simple_value=e_l)
                    summary.value.add(tag='Losses/Grad Norm', simple_value=g_n)
                    summary.value.add(tag='Losses/Var Norm', simple_value=v_n)
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                if episode_count == 120000:  # thread to stop
                    print("Stop training name:{}".format(self.name))
                    coord.request_stop()

    def play_game(self, sess, episode_num):
        if not isinstance(sess, tf.Session):
            raise TypeError('saver should be tf.train.Saver')

        for i in range(episode_num):

            self.env.new_episode()
            state = self.env.get_state()
            s = utils.process_frame(state.screen_buffer, cfg.img_dim)
            episode_rewards = 0
            last_total_shaping_reward = 0
            step = 0
            s_t = time.time()
            while not self.env.is_episode_finished():
                state = self.env.get_state()
                s = utils.process_frame(state.screen_buffer, cfg.img_dim)
                a_dist, v = sess.run([self.local_AC_network.policy, self.local_AC_network.value],
                                     feed_dict={self.local_AC_network.inputs: [s]})
                # get a action_index from a_dist in self.local_AC.policy
                a_index = self.choose_action_index(a_dist[0], deterministic=True)
                # make an action
                self.env.make_action(self.actions[a_index])

                step += 1

                shaping_reward = doom_fixed_to_double(self.env.get_game_variable(GameVariable.USER1)) / 100.
                r = (shaping_reward - last_total_shaping_reward)
                last_total_shaping_reward += r

                episode_rewards += r

                print('Current step: #{}'.format(step))
                print('Current action: ', self.actions[a_index])
                print('Current health: ', self.env.get_game_variable(GameVariable.HEALTH))
                print('Current shaping: {0}'.format(shaping_reward))
                print('Current reward: {0}'.format(r))
            print('End episode: {}, Total Reward: {}, {}'.format(i, episode_rewards, last_total_shaping_reward))
            print('time costs: {}'.format(time.time() - s_t))
            time.sleep(5)

    @staticmethod
    def choose_action_index(policy, deterministic=False):
        if deterministic:
            return np.argmax(policy)

        r = random.random()
        cumulative_reward = 0
        for i, p in enumerate(policy):
            cumulative_reward += p
            if r <= cumulative_reward:
                return i

        return len(policy) - 1

    def reward_function(self, picked_delta, episode_step_count):
        health = self.env.get_game_variable(GameVariable.HEALTH)
        if picked_delta == 0:
            # if run out of health before episode ends, give a punishment
            if health < 4 and episode_step_count < self.env.get_episode_timeout():
                reward = -1
            else:
                # alive
                reward = 0.01
        else:
            return picked_delta * 2.
        return reward


class Game(DoomGame):
    def __init__(self, scenarios_path, play):
        super(Game, self).__init__()
        self.scenarios_path = scenarios_path
        self.game_config(play)
        self.actions = []

    def init_game(self):
        self.init()

    def game_config(self, play):
        self.set_doom_scenario_path(cfg.SCENARIO_PATH)
        self.set_doom_map("map01")
        self.set_screen_resolution(ScreenResolution.RES_640X480)
        self.set_screen_format(ScreenFormat.RGB24)
        self.set_render_hud(False)
        self.set_render_crosshair(False)
        self.set_render_weapon(True)
        self.set_render_decals(False)
        self.set_render_particles(True)
        self.set_labels_buffer_enabled(True)
        self.add_available_button(Button.TURN_LEFT)
        self.add_available_button(Button.TURN_RIGHT)
        self.add_available_button(Button.MOVE_FORWARD)
        self.add_available_game_variable(GameVariable.USER1)
        self.set_episode_timeout(2100)
        self.set_episode_start_time(5)
        self.set_window_visible(play)
        self.set_sound_enabled(False)
        self.set_living_reward(0)
        self.set_mode(Mode.PLAYER)

    def get_actions(self):
        self.actions = [list(perm) for perm in it.product([False, True], repeat=self.get_available_buttons_size())]
        self.actions.remove([True, True, True])
        self.actions.remove([True, True, False])
        return self.actions
