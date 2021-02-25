import torch
from torch.nn import MSELoss
from torch.optim import Adam
import gym
import gym_pull
import ppaquette_gym_super_mario
from gym.monitoring import Monitor
import random
import numpy as np
import os
from mario_model import MarioModel
import random


class GameAsTorch:
    def __init__(self, episode_count: int = 2000, eps: float = 1.0, eps_min: float = 1e-4, stdout_interval: int = 60):
        self.episode_count = episode_count
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = (eps - eps_min) / episode_count
        self.stdout_interval = stdout_interval

        # ステージを選択
        self.env: ppaquette_gym_super_mario.SuperMarioBrosEnv = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')

        # actionは[up, left, down, right, A, B]の6種
        self.action_list = [
            # 右移動
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 1],
        ]
        return

    def create_model(self):
        model = MarioModel(action_size=len(self.action_list))
        if torch.cuda.is_available():
            model = model.to('cuda')
            pass

        return model

    def run(self, use_previous_params: bool = False):
        MODEL_DIR = "./pth_models"
        MODEL_FILENAME = "mario.pth"
        MODEL_PATH = "{0}/{1}".format(MODEL_DIR.rstrip('/'), MODEL_FILENAME)

        model = self.create_model()
        if use_previous_params:
            model.load(MODEL_PATH)
            pass

        criterion = MSELoss()
        optimizer = Adam(model.parameters(), 1e-4)
        model.train()
        total_max_distance = 0
        for episode in range(self.episode_count):
            # 初期化
            self.env.reset()
            total_score: int = 0
            distance: int = 0
            is_finished: bool = False
            frame_count: int = 0
            self.eps = max(self.eps - self.eps_decay, self.eps_min)
            while not is_finished:
                # 入力データを取得
                screen = torch.reshape(torch.from_numpy(self.env.tiles.copy()), (1, 1, 13, 16))
                # screen = torch.zeros([1, 1, 13, 16])
                if torch.cuda.is_available():
                    screen = screen.to('cuda')
                    pass

                # 推論を行う
                with torch.no_grad():
                    predicted_rewards: torch.Tensor = model(screen)
                    pass

                # 行動を決定
                if np.random.random() < self.eps:
                    action_index = random.randint(0, len(self.action_list) - 1)
                    is_random_action = True
                    pass
                else:
                    action_index = np.argmax(predicted_rewards.to('cpu').detach().numpy().copy()[0])
                    is_random_action = False
                    pass

                # ゲームを進行
                obs, reward, is_finished, info = self.env.step(self.action_list[action_index])

                # アクションの入力データ作成
                action_array = torch.zeros(len(self.action_list))
                action_array[action_index] = 1

                # 報酬を算出
                movement_reward: float = 0
                if info['distance'] > distance:
                    movement_reward = float(0.5)
                    pass
                else:
                    movement_reward = float(-0.1)
                    pass
                distance = info['distance']

                reward_truth = torch.Tensor([movement_reward])

                # 学習を行う
                _reward_predict = torch.matmul(predicted_rewards, torch.transpose(torch.unsqueeze(action_array, 0), 0, 1))
                _reward_predict.requires_grad = True
                reward_predict = torch.sum(_reward_predict, 1)
                optimizer.zero_grad()
                loss = criterion(reward_truth, reward_predict)
                loss.backward()
                optimizer.step()

                # 情報の出力
                if frame_count % self.stdout_interval is 0:
                    print("Episode: {0}, eps: {1}".format(episode, self.eps))
                    print("is_random_action: {0}".format(is_random_action))
                    print("reward_predict: {0}, reward_truth: {1}".format(reward_predict, reward_truth))
                    print("loss: ", loss.item())
                    print("info: {0}".format(info))
                    print()
                    pass

                # 毎フレームの更新処理
                self.env.render()
                frame_count += 1

                pass

            # エピソードが終了する度に行う処理
            model.save()

        return


if __name__ == '__main__':
    game = GameAsTorch()
    game.run()
