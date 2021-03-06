import gym
import gym_network

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class NetworkSolver():
    def __init__(self, n_episodes=10000, gamma=0.5, epsilon=0.68, epsilon_min=0.01, epsilon_log_decay=0.5, lr=0.001, lr_decay=0.01, batch_size=64):
        self.memory = deque(maxlen=1024 * 512)
        self.env = gym.make('Network-v0')
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.lr = lr
        self.lr_decay = lr_decay
        self.n_episodes = n_episodes
        self.batch_size = batch_size

        #remember, this may have to be different for network.
        # Init model
        self.model = Sequential()
        self.model.add(Dense(8, input_dim=4, activation='relu'))
        #self.model.add adds a layer to the neural network. Only the first self.model.add takes a input_dim
        # self.model.add(Dense(24, activation='relu'))  # 48 tanh
        self.model.add(Dense(16, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam())  # lr=self.lr, decay=self.lr_decay))
        # try different loss and optimizers and models

        # results
        self.success = deque(maxlen=n_episodes)  # will contain [ep_number, duration, tot_reward]
        plt.ion()
        plt.figure(1)

    def show(self):
        """ shows training related variables for each episode """
        df = pd.DataFrame(list(self.success))
        print(df.head())

        plt.subplot(511)
        plt.plot(df[1])
        plt.yscale('log')
        plt.title('duration')

        plt.subplot(512)
        plt.plot(df[2])
        plt.title('reward0')
        
        plt.subplot(513)
        plt.plot(df[3])
        plt.title('reward1')
        
        plt.subplot(514)
        plt.plot(df[4])
        plt.title('reward2')
        
        plt.subplot(515)
        plt.plot(df[5])
        plt.title('reward3')

        plt.pause(0.001)
        plt.show()

    def remember(self, state, action, reward, next_state, done):
        """ just adds to memory deque """
        self.memory.append((state, action, reward, next_state, done))
        
    def max_array(self, prediction):
        target_arrays = [prediction[i*4 : (i*4)+4] for i in range(4)]
        max_array1 = []
        for a in target_arrays:
            max_array1.append(np.argmax(a))
        return max_array1

    def choose_action(self, state, epsilon):
        """ """
        # return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else self.max_array(self.model.predict(state)[0])

    def get_epsilon(self, e):
        """  epsilon will depend on the episode.
        Higher the epsilon - higher the chance to randomly select action.
        a) it will be at least eposilon_min.
        b) it will be at most epsilon
        c) with epoch number epsilon will decrease
            * epsilon_decay=1.0 will reach epsilon_min at <=10 episodes
            * epsilon_decay=0.5 will reach epsilon_min at <=20
            * epsilon_decay=0.1 will reach epsilon_min at <=100
            * epsilon_decay=0.01 will reach epsilon_min at <=1000
        """
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((e + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        print("This is the state: ", state)
        return np.reshape(state, (1, 4))
        # return np.reshape(state, [1, 3])

    def replay(self, batch_size):
        """  It replays a part of the hystoric data and learns on it.
        One or more batches. Called once AFTER each episode.
        """
        nbatches = len(self.memory) // self.batch_size
        nbatches = min(nbatches, 1)
        for n in range(nbatches):
            # print('batch', n)
            # print('.', end='')
            x_batch, y_batch = [], []
            # x_batch = a list of observation values; y_batch = a list of np array of 4 rewards (corresponding to each potential action)
            minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
            for state, action, reward, next_state, done in minibatch:
                
                # y_target = [[v0, v1, ..., v15]], y_target[0] = [v0, v1, ..., v15]
                # state is something like : [[ 1.33765948  1.44549417  0.57516229  1.1875875 ]], 
                # state[0] = [ 1.33765948  1.44549417  0.57516229  1.1875875 ]
                
                y_target = self.model.predict(state)
                
                target_arrays = [y_target[0][i*4 : (i*4)+4] for i in range(4)]
                max_array = []
                for a in target_arrays:
                    max_array.append(np.max(a))
                    
                print("------------ACTION2 = ", action)
                
                if done:
                    # y_target[0][action] = reward
                    print("action[0] = ", action[0])
                    for i in range(4):
                        y_target[0][i*4 + action[i]] = reward[i]
                else:
                    for i in range(4):
                        y_target[0][i*4 + action[i]] = reward[i] + self.gamma * max_array[i]
                        
                    # y_target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
                    # y_target[0] = reward + self.gamma * self.model.predict(next_state)[0]
                    #Multiplying by gamma accounts for discount ratio
                x_batch.append(state[0])
                y_batch.append(y_target[0])

            hist = self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), epochs=100, verbose=0)
            #print("This is loss: ", hist.history['loss'])
            #print("This is hist ", hist)
        # print('.')

    def run(self):
        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            tot_reward = [0.0, 0.0, 0.0, 0.0]
            steps = 0
            info = None
            while not done:
                action = self.choose_action(state, self.get_epsilon(e))
                print("action = ", action)
                next_state, reward, done, info = self.env.step(action)
           
                if e % 50 == 0:
                    self.env.render()
                    print("next_state length: ", len(next_state))
                    print("reward length: ", len(reward))
                    print(action, 'state:{: 2.5f}, {: 2.5f}, {: 2.5f}, {: 2.5f}   rew:{: 2.5f}, {: 2.5f}, {: 2.5f}, {: 2.5f}'.format(next_state[0], next_state[1], next_state[2], next_state[3], reward[0], reward[1], reward[2], reward[3]))
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                for i in range(4):
                    tot_reward[i] += reward[i]
                steps += 1
                # if steps % 200 == 0:
                #    print('episode:', e, '\tsteps:', steps)

            self.success.append([e, steps, tot_reward[0], tot_reward[1], tot_reward[2], tot_reward[3]])
            if e % 50 == 0:
                self.show()

            print('episode:{:4d}  duration:{:7d}    reward:{: 9.2f}, {: 9.2f}, {: 9.2f}, {: 9.2f}'.format(e, steps, tot_reward[0], tot_reward[1], tot_reward[2], tot_reward[3]), info)

            self.replay(self.batch_size)
        return


if __name__ == '__main__':
    agent = NetworkSolver()
    agent.run()
