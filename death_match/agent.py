# coding: utf-8
# implement of agent
import time
import random

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
    def __init__(self, game, name, optimizer=None, model_path=None, global_episodes=None, play=False, img_shape=(80, 80)):
        self.img_shape = img_shape
        self.summary_step = 3

        self.name = "worker_" + str(name)
        self.number = name

        self.episode_reward = []
        self.episode_episode_total_pickes = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.episode_health = []
        self.episode_kills = []
        self.last_weapon_ammo_dict = {1: 0, 2: 50, 3: 48, 4: 200, 5: 27, 6: 300}
        self.last_total_health = 100.
        self.last_total_kills = 0
        self.last_total_reward = 0
        self.last_position_xyz = [0, 0, 0]

        # Create the local copy of the network and the tensorflow op to
        # copy global parameters to local network
        if not play:
            self.model_path = model_path
            self.trainer = optimizer
            self.global_episodes = global_episodes
            self.increment = self.global_episodes.assign_add(1)
            self.local_AC_network = network.ACNetwork(self.name, optimizer, play=play, shape=self.img_shape)
            self.summary_writer = tf.summary.FileWriter("./summaries/death_match/agent_%s" % str(self.number))
            self.update_local_ops = tf.group(*utils.update_target_graph('global', self.name))
        else:
            self.local_AC_network = network.ACNetwork(self.name, optimizer, play=play, shape=self.img_shape)
        if not isinstance(game, DoomGame):
            raise TypeError("Type Error")

        # The Below code is related to setting up the Doom environment
        game = DoomGame()
        # game.set_doom_scenario_path('../scenarios/deadly_corridor.cfg')
        game.load_config("../scenarios/deathmatch.cfg")
        # game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.set_screen_format(ScreenFormat.RGB24)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        # Enables labeling of the in game objects.
        game.set_labels_buffer_enabled(True)
        game.add_available_button(Button.ATTACK)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_FORWARD)
        game.add_available_button(Button.TURN_RIGHT)
        game.add_available_button(Button.TURN_LEFT)
        game.set_episode_timeout(4200)
        game.set_episode_start_time(1)
        game.set_window_visible(play)
        game.set_sound_enabled(False)
        game.set_living_reward(0)
        game.set_mode(Mode.PLAYER)
        if play:
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
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
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
            self.local_AC_network.actions: actions,
            self.local_AC_network.advantages: advantages,
            self.local_AC_network.game_variables: game_vars
        }
        l, v_l, p_l, e_l, g_n_v, g_n_p, v_n, _, _ = sess.run([
                                            self.local_AC_network.loss,
                                            self.local_AC_network.value_loss,
                                            self.local_AC_network.policy_loss,
                                            self.local_AC_network.entropy,
                                            self.local_AC_network.grad_norms_value,
                                            self.local_AC_network.grad_norms_policy,
                                            self.local_AC_network.var_norms,
                                            self.local_AC_network.apply_grads_value,
                                            self.local_AC_network.apply_grads_policy],
                                            feed_dict=feed_dict)
        return l / len(rollout), v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n_v, g_n_p, v_n

    def train_a3c(self, max_episode_length, gamma, sess, coord, saver):
        if not isinstance(saver, tf.train.Saver):
            raise TypeError('saver should be tf.train.Saver')

        episode_count = sess.run(self.global_episodes)
        start_t = time.time()
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)  # update local ops in every episode
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                terminate = False

                self.env.new_episode()
                self.last_position_xyz = self.get_position_xyz()
                st_s = utils.process_frame(self.env.get_state().screen_buffer, self.img_shape)
                s = np.stack((st_s, st_s, st_s, st_s), axis=2)
                episode_st = time.time()

                while not self.env.is_episode_finished():
                    # get game variables before taking an action
                    game_vars = self.get_game_variables()
                    # Take an action using probabilities from policy network output.
                    a_dist, v = sess.run([self.local_AC_network.policy, self.local_AC_network.value],
                                         feed_dict={self.local_AC_network.inputs: [s],
                                                    self.local_AC_network.game_variables: [game_vars]})
                    # get a action_index from a_dist in self.local_AC.policy
                    a_index = self.choose_action_index(a_dist[0], deterministic=False)
                    # make an action
                    self.env.make_action(self.actions[a_index], cfg.SKIP_FRAME_NUM) / float(cfg.SKIP_FRAME_NUM)
                    reward = self.reward_function()
                    episode_reward += reward

                    terminate = self.env.is_episode_finished()
                    if terminate:
                        s1 = s
                    else:  # game is not finished
                        img = np.reshape(utils.process_frame(
                            self.env.get_state().screen_buffer, self.img_shape), (*self.img_shape, 1))
                        s1 = np.append(img, s[:, :, :3], axis=2)

                    episode_buffer.append([s, a_index, reward, s1, terminate, v[0, 0], game_vars])
                    episode_values.append(v[0, 0])
                    # summaries information
                    s = s1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == cfg.BUFFER_SIZE and terminate is False \
                            and episode_step_count != max_episode_length - 1:
                        v1 = sess.run(self.local_AC_network.value,
                                      feed_dict={self.local_AC_network.inputs: [s],
                                                 self.local_AC_network.game_variables: [game_vars]})[0, 0]
                        l, v_l, p_l, e_l, g_n_v, g_n_p, v_n = self.infer(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if terminate is True:
                        self.episode_health.append(self.env.get_game_variable(GameVariable.HEALTH))
                        print('{}, health: {}, kills:{}, episode #{}, reward: {}, steps:{}, time costs:{}'.format(
                            self.name, self.last_total_health, self.last_total_kills, episode_count,
                            episode_reward, episode_step_count, time.time()-episode_st))
                        break

                # summaries
                self.episode_kills.append(self.last_total_kills)
                self.episode_reward.append(episode_reward)
                self.episode_episode_total_pickes.append(self.last_total_health)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    l, v_l, p_l, e_l, g_n_v, g_n_p, v_n = self.infer(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 50 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path+'/model-'+str(episode_count)+'.ckpt')
                        print("Episode count {}, saved Model, time costs {}".format(episode_count, time.time()-start_t))
                        start_t = time.time()

                    mean_picked = np.mean(self.episode_episode_total_pickes[-5:])
                    mean_reward = np.mean(self.episode_reward[-5:])
                    mean_health = np.mean(self.episode_health[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    mean_kills = np.mean(self.episode_kills[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Performance/Reward', simple_value=mean_reward)
                    summary.value.add(tag='Performance/Health', simple_value=mean_health)
                    summary.value.add(tag='Performance/Kills', simple_value=mean_kills)
                    summary.value.add(tag='Performance/Steps', simple_value=mean_length)
                    summary.value.add(tag='Performance/Value', simple_value=mean_value)
                    summary.value.add(tag='Losses/Total Loss', simple_value=l)
                    summary.value.add(tag='Losses/Value Loss', simple_value=v_l)
                    summary.value.add(tag='Losses/Policy Loss', simple_value=p_l)
                    summary.value.add(tag='Losses/Entropy', simple_value=e_l)
                    summary.value.add(tag='Losses/Grad Norm value', simple_value=g_n_v)
                    summary.value.add(tag='Losses/Grad Norm policy', simple_value=g_n_p)
                    summary.value.add(tag='Losses/Var Norm', simple_value=v_n)
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                if episode_count == 12000000:  # thread to stop
                    print("Stop training name:{}".format(self.name))
                    coord.request_stop()

    def play_game(self, sess, episode_num):
        if not isinstance(sess, tf.Session):
            raise TypeError('saver should be tf.train.Saver')

        for i in range(episode_num):

            self.env.new_episode()
            st_s = utils.process_frame(self.env.get_state().screen_buffer, self.img_shape)
            s = np.stack((st_s, st_s, st_s, st_s), axis=2)
            episode_rewards = 0
            last_total_shaping_reward = 0
            step = 0
            s_t = time.time()

            while not self.env.is_episode_finished():
                game_vars = self.get_game_variables()
                a_dist, v = sess.run([self.local_AC_network.policy, self.local_AC_network.value],
                                     feed_dict={self.local_AC_network.inputs: [s],
                                                self.local_AC_network.game_variables: [game_vars]})

                img = utils.process_frame(self.env.get_state().screen_buffer, self.img_shape)
                img = np.reshape(img, (*self.img_shape, 1))
                s = np.append(img, s[:, :, :3], axis=2)

                # get a action_index from a_dist in self.local_AC.policy
                a_index = self.choose_action_index(a_dist[0], deterministic=True)
                # make an action
                reward = self.env.make_action(self.actions[a_index])

                step += 1

                shaping_reward = doom_fixed_to_double(self.env.get_game_variable(GameVariable.USER1)) / 100.
                r = (shaping_reward - last_total_shaping_reward)
                last_total_shaping_reward += r

                episode_rewards += reward
                print('Current weapon: {}, type: {}'.format(self.env.get_game_variable(GameVariable.SELECTED_WEAPON),
                                                            type(self.env.get_game_variable(GameVariable.SELECTED_WEAPON))))
                print('Current step: #{}'.format(step))
                print('Current action: ', self.actions[a_index])
                print('Current health: ', self.env.get_game_variable(GameVariable.HEALTH))
                print('Current kills: {0}'.format(self.env.get_game_variable(GameVariable.KILLCOUNT)))
                print('Current reward: {0}'.format(reward))
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

    def get_game_variables(self):
        return np.array([
            self.env.get_game_variable(GameVariable.HEALTH),
            self.env.get_game_variable(GameVariable.AMMO2),
            self.env.get_game_variable(GameVariable.AMMO3),
            self.env.get_game_variable(GameVariable.AMMO4),
            self.env.get_game_variable(GameVariable.AMMO5),
            self.env.get_game_variable(GameVariable.AMMO6)
        ], dtype=np.float32)

    def __kills_reward(self):
        kill_count = self.env.get_game_variable(GameVariable.KILLCOUNT)
        kill_delta = kill_count - self.last_total_kills
        reward = 0
        if kill_delta > 0:
            reward = kill_delta
        return reward, kill_count

    def __weapon_reward(self):
        current_weapon = self.env.get_game_variable(GameVariable.SELECTED_WEAPON)
        if current_weapon == 1.0:
            return -1
        elif current_weapon == 2.0:
            return -0.01
        elif current_weapon == 3.0:
            return .5
        elif current_weapon == 4.0:
            return .5
        elif current_weapon == 5.0:
            return 1.
        elif current_weapon == 6.0:
            return 1.5

    def reward_function(self):
        # living reward
        living_reward = self.__penalize_living()
        # kill reward
        kill_reward, self.last_total_kills = self.__kills_reward()
        # ammo2 reward
        ammo2_reward = self.__ammo_reward()
        # health reward
        health_reward = self.__health_reward()
        # weapon reward
        weapon_reward = self.__weapon_reward()
        # distance reward
        distance_reward = self.__distance_reward()
        reward = living_reward + kill_reward + health_reward + ammo2_reward + distance_reward + weapon_reward
        return reward

    def reward_function_2(self):
        reward = self.env.get_total_reward() - self.last_total_reward
        dist_reward = self.__distance_reward()
        weapon_reward = self.__weapon_reward()
        self.last_total_reward = self.env.get_total_reward()
        self.last_total_health = self.env.get_game_variable(GameVariable.HEALTH)
        self.last_total_kills = self.env.get_game_variable(GameVariable.KILLCOUNT)
        return reward + dist_reward + weapon_reward

    def __distance_reward(self):
        current_position = self.get_position_xyz()
        dist = np.sqrt(sum([(v1-v2)**2 for v1, v2 in zip(current_position, self.last_position_xyz)]))
        self.last_position_xyz = self.get_position_xyz()
        if dist <= cfg.STAY_PENALIZE_THRESHOLD_VALUE:
            return -0.03  # -0.03 * cfg.SKIP_FRAME_NUM
        else:
            return 9e-5 * dist

    @staticmethod
    def __penalize_living():
        return -0.008

    def __health_reward(self):
        health_delta = self.env.get_game_variable(GameVariable.HEALTH) - self.last_total_health
        self.last_total_health = self.env.get_game_variable(GameVariable.HEALTH)
        if health_delta < 0:
            return -0.05 * health_delta
        else:
            return 0.04 * health_delta

    def __ammo_reward(self):
        ammo_delta = self.env.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO) - \
            self.last_weapon_ammo_dict[int(self.env.get_game_variable(GameVariable.SELECTED_WEAPON))]
        self.last_weapon_ammo_dict[int(self.env.get_game_variable(GameVariable.SELECTED_WEAPON))] = \
            self.env.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)

        if ammo_delta >= 0:
            return 0.15 * ammo_delta
        elif ammo_delta < 0:
            return -0.04 * ammo_delta

    def get_position_xyz(self):
        return [
            self.env.get_game_variable(GameVariable.POSITION_X),
            self.env.get_game_variable(GameVariable.POSITION_Y),
            self.env.get_game_variable(GameVariable.POSITION_Z)
        ]
