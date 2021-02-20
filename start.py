import tensorflow as tf
import gym
import gym_pull
import ppaquette_gym_super_mario
from gym.monitoring import Monitor
import random
import numpy as np


class Game:

    def __init__(self, episode_count=2000, eps=1.0, eps_min=1e-4):
        self.episode_count = episode_count
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = (eps - eps_min) / episode_count
        # select stage
        self.env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')

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
        action_list = [[0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 0]]

        sess = tf.compat.v1.InteractiveSession()
        s, readout = self.create_network(len(action_list))
        a = tf.compat.v1.placeholder("float", [None, len(action_list)])
        y = tf.compat.v1.placeholder("float", [None, 1])
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        train_step = tf.compat.v1.train.AdamOptimizer(1e-6).minimize(cost)

        saver = tf.compat.v1.train.Saver()
        sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("./saved_networks/checkpoints")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        for episode in range(self.episode_count):
            self.env.reset()
            total_score = 0
            distance = 0
            is_finished = False
            actions, rewards, images = [], [], []
            self.eps = max(self.eps - self.eps_decay, self.eps_min)
            while is_finished == False:
                screen = np.reshape(self.env.tiles, (13, 16, 1))
                readout_t = readout.eval(feed_dict={s: [screen]})[0]

                print("eps: {0}".format(self.eps))
                if np.random.random() < self.eps:
                    action_index = random.randint(0, len(action_list) - 1)
                else:
                    action_index = np.argmax(readout_t)

                # 画面のマリオに処理する
                obs, reward, is_finished, info = self.env.step(action_list[action_index])
                print("info: {}".format(info))

                # MiniBatch化するように配列に
                action_array = np.zeros(len(action_list))
                action_array[action_index] = 1
                actions.append(action_array)

                # 報酬を与える
                rewards.append([float(info['distance'])])

                images.append(screen)

                train_step.run(feed_dict={
                    a: actions, y: rewards, s: images
                })
                print("Episode: {0}, Actions: {1}, Rewards: {2}, readout_t: {3}\n".format(
                    episode, action_index, rewards, readout_t)
                )
                actions, rewards, images = [], [], []

                self.env.render()
            saver.save(sess, 'saved_networks/model-dqn', global_step=episode)


if __name__ == '__main__':
    game = Game()
    game.play_game()
