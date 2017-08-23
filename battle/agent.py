# coding: utf-8
# implement of agent
import time
import random
import pickle
import itertools as iter

import numpy as np
import tensorflow as tf

from vizdoom import *

import utils
import network
import configs as cfg


class Agent(object):
    """
    Agent
    """
    def __init__(self, game, name, optimizer=None, model_path=None,
                 global_episodes=None, play=False, task_name='healthpack_simple'):
        self.task_name = task_name
        self.play = play
        self.summary_step = 3

        self.visualizer = utils.Visualiser()

        self.name = cfg.AGENT_PREFIX + str(name)
        self.number = name

        self.last_total_health = 100.
        self.last_total_kills = 0.
        self.last_total_ammos = 0.
        self.img_shape = cfg.IMG_SHAPE

        self.episode_reward = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.episode_health = []
        self.episode_kills = []

        if not self.play:
            self.model_path = model_path
            self.trainer = optimizer
            self.global_episodes = global_episodes
            self.increment = self.global_episodes.assign_add(1)
            self.local_AC_network = network.ACNetwork(self.name, optimizer, play=self.play, img_shape=cfg.IMG_SHAPE)
            self.summary_writer = tf.summary.FileWriter("./summaries/%s/ag_%s" % (self.task_name, str(self.number)))
            # create a tensorflow op to copy weights from global network regularly when training
            self.update_local_ops = tf.group(*utils.update_target_graph('global', self.name))
        else:
            self.local_AC_network = network.ACNetwork(self.name, optimizer, play=self.play, img_shape=cfg.IMG_SHAPE)
        if not isinstance(game, DoomGame):
            raise TypeError("Type Error")

        game = DoomGame()
        game.load_config(cfg.SCENARIO_PATH)
        game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.set_screen_format(ScreenFormat.RGB24)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(True)
        # Enables labeling of the in game objects.
        game.set_labels_buffer_enabled(True)
        game.add_available_button(Button.MOVE_FORWARD)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.TURN_LEFT)
        game.add_available_button(Button.TURN_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_button(Button.SPEED)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.HEALTH)
        game.add_available_game_variable(GameVariable.USER2)
        game.set_episode_timeout(2100)
        game.set_episode_start_time(5)
        game.set_window_visible(self.play)
        game.set_sound_enabled(False)
        game.set_living_reward(0)
        game.set_mode(Mode.PLAYER)
        if self.play:
            game.add_game_args("+viz_render_all 1")
            game.set_render_hud(False)
            game.set_ticrate(35)
        game.init()
        self.env = game
        self.actions = cfg.button_combinations()

    def infer(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]
        game_vars = np.array([v.tolist() for v in rollout[:, 6].tolist()], dtype=np.float32)
        # The advantage function uses "Generalized Advantage Estimation"
        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = utils.discount(rewards_plus, gamma)[:-1]
        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
        advantages = utils.discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {
            self.local_AC_network.target_v: discounted_rewards,
            self.local_AC_network.inputs: np.stack(observations),
            self.local_AC_network.game_variables: game_vars,
            self.local_AC_network.actions: actions,
            self.local_AC_network.advantages: advantages
        }
        l, v_l, p_l, e_l, g_n, v_n, _ = sess.run([
                                            self.local_AC_network.loss,
                                            self.local_AC_network.value_loss,
                                            self.local_AC_network.policy_loss,
                                            self.local_AC_network.entropy,
                                            self.local_AC_network.grad_norms,
                                            self.local_AC_network.var_norms,
                                            self.local_AC_network.apply_grads],
                                            feed_dict=feed_dict)
        return l / len(rollout), v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def train_a3c(self, max_episode_length, gamma, sess, coord, saver):
        if not isinstance(saver, tf.train.Saver):
            raise TypeError('saver should be tf.train.Saver')

        episode_count = sess.run(self.global_episodes)
        start_t = time.time()
        print("Starting  %s%s" % (cfg.AGENT_PREFIX, str(self.number)))
        self.last_total_ammos = self.env.get_game_variable(GameVariable.AMMO2)

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)  # update local ops in every episode
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                end = False

                self.env.new_episode()
                st_s = utils.process_frame(self.env.get_state().screen_buffer, self.img_shape)
                s = np.stack((st_s, st_s, st_s, st_s), axis=2)
                episode_st = time.time()

                while not self.env.is_episode_finished():
                    game_vars = self.env.get_state().game_variables[:-1]
                    state = [s, game_vars]

                    reward, v, end, a_index = self.step(state, sess)

                    episode_reward += reward

                    if end:
                        s1 = s
                    else:  # game is not finished
                        img = np.reshape(utils.process_frame(
                            self.env.get_state().screen_buffer, self.img_shape), (*self.img_shape, 1))
                        s1 = np.append(img, s[:, :, :3], axis=2)

                    episode_buffer.append([s, a_index, reward, s1, end, v[0, 0], game_vars])
                    episode_values.append(v[0, 0])
                    # summaries information
                    s = s1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 64 and end is False and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is,
                        # we "bootstrap" from our current value estimation.
                        v1 = sess.run(self.local_AC_network.value, feed_dict={
                                      self.local_AC_network.inputs: [s],
                                      self.local_AC_network.game_variables: [self.env.get_state().game_variables[:-1]]
                        })[0, 0]
                        l, v_l, p_l, e_l, g_n, v_n = self.infer(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                    if end is True:
                        self.episode_health.append(self.env.get_game_variable(GameVariable.HEALTH))
                        self.episode_kills.append(self.env.get_game_variable(GameVariable.USER2))
                        print('{}, health: {}, kills: {}, episode #{}, reward: {}, steps:{}, time costs:{}'.format(
                            self.name, self.env.get_game_variable(GameVariable.HEALTH),
                            self.env.get_game_variable(GameVariable.USER2), episode_count,
                            episode_reward, episode_step_count, time.time()-episode_st))
                        break

                # summaries
                self.episode_reward.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    l, v_l, p_l, e_l, g_n, v_n = self.infer(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 50 == 0 and self.name == cfg.AGENT_MONITOR:
                        saver.save(sess, self.model_path+'/model-'+str(episode_count)+'.ckpt')
                        print("Episode count {}, saved Model, time costs {}".format(episode_count, time.time()-start_t))
                        start_t = time.time()

                    mean_reward = np.mean(self.episode_reward[-5:])
                    mean_health = np.mean(self.episode_health[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    mean_kills = np.mean(self.episode_kills[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Performance/Reward', simple_value=mean_reward)
                    summary.value.add(tag='Performance/Kills', simple_value=mean_kills)
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

                if self.name == cfg.AGENT_MONITOR:
                    sess.run(self.increment)
                episode_count += 1
                if episode_count == 120000:  # thread to stop
                    print("Stop training name:{}".format(self.name))
                    coord.request_stop()

    def play_game(self, sess, episode_num):
        if not isinstance(sess, tf.Session):
            raise TypeError('saver should be tf.train.Saver')

        self.visualizer.init()

        for i in range(episode_num):

            self.env.new_episode()

            st_s = utils.process_frame(self.env.get_state().screen_buffer, self.img_shape)
            s = np.stack((st_s, st_s, st_s, st_s), axis=2)
            episode_rewards = 0
            step = 0
            s_t = time.time()

            while not self.env.is_episode_finished():
                game_vars = self.env.get_state().game_variables
                state = [s, game_vars[:-1]]

                reward, v, end, a_index = self.step(state, sess)
                self.visualizer.visualize(s, game_vars, self.actions[a_index], reward, v)
                if end:
                    break

                img = np.reshape(
                    utils.process_frame(self.env.get_state().screen_buffer, self.img_shape), (*self.img_shape, 1))
                s = np.append(img, s[:, :, :3], axis=2)

                step += 1

                episode_rewards += reward

                print('Current step: #{}'.format(step))
                print('Current action: ', self.actions[a_index])
                print('Current game variables: ', self.env.get_state().game_variables)
                print('Current reward: {0}'.format(reward))
                time.sleep(0.05)
            print('End episode: {}, Total Reward: {}'.format(i, episode_rewards))
            print('time costs: {}'.format(time.time() - s_t))
            time.sleep(5)

    def step(self, state, sess):
        if not isinstance(sess, tf.Session):
            raise TypeError('TypeError')

        s, game_vars = state
        a_dist, value = sess.run([self.local_AC_network.policy, self.local_AC_network.value], feed_dict={
            self.local_AC_network.inputs: [s],
            self.local_AC_network.game_variables: [game_vars]
        })
        a_index = self.choose_action_index(a_dist[0], deterministic=False)
        if self.play:
            self.env.make_action(self.actions[a_index])
        else:
            self.env.make_action(self.actions[a_index], cfg.SKIP_FRAME_NUM)

        reward = self.reward_function()
        end = self.env.is_episode_finished()

        return reward, value, end, a_index

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

    def reward_function(self):
        kills_delta = self.env.get_game_variable(GameVariable.USER2) - self.last_total_kills
        ammo_delta = self.env.get_game_variable(GameVariable.AMMO2) - self.last_total_ammos
        health_delta = self.env.get_game_variable(GameVariable.HEALTH) - self.last_total_health

        reward = 0
        reward += kills_delta * 20

        # health issues
        if self.last_total_health < 50:  # dangerous situation
            if health_delta > 0:
                reward += health_delta * 1.    # large reward
            else:
                reward -= health_delta * .5    # large penalty

        elif self.last_total_health <= 100:  # moderate situation
            if health_delta > 0:
                reward += health_delta * .5    # moderate reward
            else:
                reward -= health_delta * .1    # moderate penalty

        else:                               # normal situation
            if health_delta < 0:
                reward -= health_delta * .1    # moderate penalty

        # ammo issues
        if self.last_total_ammos < 10:   # dangerous situation
            if ammo_delta > 0:
                reward += ammo_delta * 2.
            else:
                reward -= ammo_delta * 1.
        elif self.last_total_ammos <= 20:  # moderate situation
            if ammo_delta > 0:
                reward += ammo_delta * 1.
            # else:
            #     reward -= ammo_delta * 0.1
        else:                             # the more the better
            if ammo_delta > 0:
                reward += ammo_delta * .5
            # else:
            #     reward -= ammo_delta * 0.05

        self.last_total_kills = self.env.get_game_variable(GameVariable.USER2)
        self.last_total_ammos = self.env.get_game_variable(GameVariable.AMMO2)
        self.last_total_health = self.env.get_game_variable(GameVariable.HEALTH)

        return reward
