import tensorflow as tf
import gym
import gym_pull
import ppaquette_gym_super_mario
from gym.monitoring import Monitor
import random
import numpy as np
import sys


class Game:

    def __init__(self, episode_count=2000, eps=1.0, eps_min=1e-4, lr: float = 1e-2, stdout_interval: int = 60):
        self.episode_count = episode_count
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = (eps - eps_min) / episode_count
        self.stdout_interval = stdout_interval
        self.lr = lr
        # select stage
        self.env: ppaquette_gym_super_mario.SuperMarioBrosEnv = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')

    def weight_variable(self, shape):
        initial = tf.random.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def create_network(self, action_size):
        s = tf.compat.v1.placeholder("float", [None, 13, 16, 1])

        # 1層目
        W_conv1 = self.weight_variable([8, 8, 1, 16])
        b_conv1 = self.bias_variable([16])
        h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 2) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        # 2層目
        W_conv2 = self.weight_variable([4, 4, 16, 32])
        b_conv2 = self.bias_variable([32])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 1) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # 3層目
        W_conv3 = self.weight_variable([4, 4, 32, 64])
        b_conv3 = self.bias_variable([64])
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3, 1) + b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)

        # 4層目
        W_conv4 = self.weight_variable([4, 4, 64, 64])
        b_conv4 = self.bias_variable([64])
        h_conv4 = tf.nn.relu(self.conv2d(h_pool3, W_conv4, 1) + b_conv4)
        h_pool4 = self.max_pool_2x2(h_conv4)

        # 5層目
        W_conv5 = self.weight_variable([4, 4, 64, 64])
        b_conv5 = self.bias_variable([64])
        h_conv5 = tf.nn.relu(self.conv2d(h_pool4, W_conv5, 1) + b_conv5)
        h_pool5 = self.max_pool_2x2(h_conv5)

        # 6層目
        W_conv6 = self.weight_variable([4, 4, 64, 64])
        b_conv6 = self.bias_variable([64])
        h_conv6 = tf.nn.relu(self.conv2d(h_pool5, W_conv6, 1) * b_conv6)
        h_pool6 = self.max_pool_2x2(h_conv6)

        h_conv6_flat = tf.reshape(h_pool6, [-1, 8])
        W_fc1 = self.weight_variable([8, action_size])
        b_fc1 = self.bias_variable([action_size])
        readout = tf.matmul(h_conv6_flat, W_fc1) + b_fc1
        return s, readout

    def play_game(self):
        # actionは[up, left, down, right, A, B]の6種
        action_list = [
            # 右移動
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 0],
        ]

        sess = tf.compat.v1.InteractiveSession()
        s, readout = self.create_network(len(action_list))
        a = tf.compat.v1.placeholder("float", [None, len(action_list)])
        y = tf.compat.v1.placeholder("float", [None, 1])
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        train_step = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(cost)

        saver = tf.compat.v1.train.Saver()
        sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("./saved_networks/checkpoints")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        total_max_distance = 0
        for episode in range(self.episode_count):
            self.env.reset()
            total_score = 0
            distance = 0
            time_limit = sys.maxsize
            is_finished = False
            actions, rewards, images = [], [], []
            frame_count = 0
            self.eps = max(self.eps - self.eps_decay, self.eps_min)
            while is_finished == False:
                screen = np.reshape(self.env.tiles, (13, 16, 1))
                readout_t: np.ndarray = readout.eval(feed_dict={s: [screen]})[0]

                if np.random.random() < self.eps:
                    action_index = random.randint(0, len(action_list) - 1)
                    is_random_action = True
                else:
                    action_index = np.argmax(readout_t)
                    is_random_action = False

                # 画面のマリオに処理する
                obs, reward, is_finished, info = self.env.step(action_list[action_index])

                # MiniBatch化するように配列に
                action_array = np.zeros(len(action_list))
                action_array[action_index] = 1
                actions.append(action_array)

                # 報酬を与える
                movement_reward = 0  # 移動報酬
                if info['distance'] > distance:
                    movement_reward = float(100)
                elif info['distance'] < distance:
                    movement_reward = float(-100)
                    pass
                distance = info['distance']
                total_max_distance = distance if info['distance'] > total_max_distance else total_max_distance

                score_reward = 0  # スコア報酬
                if info['score'] > total_score:
                    score_reward = float(10)
                    total_score = info['score']
                    pass

                time_reward = 0  # 残り時間報酬
                if info['time'] < time_limit:
                    time_reward = float(-1)
                    time_limit = info['time']

                composited_reward = movement_reward + score_reward + time_reward
                rewards.append([composited_reward])

                # 環境
                images.append(screen)

                # 学習を行う
                train_step.run(feed_dict={
                    a: actions, y: rewards, s: images
                })

                # 情報の出力
                if frame_count % self.stdout_interval == 0:
                    print("Episode: {0} / {1}, eps: {2}".format(episode, self.episode_count, self.eps))
                    print("is_random_action: {0}".format(is_random_action))
                    print("action_index: {0}".format(action_index))
                    print("reward_predicts: {0}".format(readout_t))
                    print("reward_predict: {0}, reward_truth: {1}".format(readout_t[action_index], composited_reward))
                    print("info: {}".format(info))
                    print("total_max_distance: {}\n".format(total_max_distance))
                    pass

                actions, rewards, images = [], [], []

                frame_count += 1
                self.env.render()
                pass

            # エピソードが終了する度に行う処理
            saver.save(sess, 'saved_networks/model-dqn', global_step=episode)
            frame_count = 0
            pass
        return


if __name__ == '__main__':
    game = Game()
    game.play_game()
