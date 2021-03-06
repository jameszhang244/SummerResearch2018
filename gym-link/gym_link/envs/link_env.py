"""
One network link environment.
Link has changing base load.
Actions: start 0 to 4 more transfers
Reward: percentage of free rate used. Gets negative if link fully saturated
Files sizes are normally distributed (absolute values).
"""

import math
from collections import deque

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np


class LinkEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.max_free_bandwidth = 0
        #max_rate is the amount of files you're adding on top of the base rate.
        self.max_rate = 0 # JZ
        self.max_link_rate = 10 * 1024 * 1024 * 1024 / 8  # 10 Gigabits - all rates are in B/s
        self.base_rate_min = 0
        self.base_rate_max = self.max_link_rate * 0.9
        self.handshake_duration = 1  # seconds
        self.max_rate_per_file = 5 * 1024 * 1024  # B/s
        self.file_size_mean = 1350 * 1024 * 1024
        self.file_size_sigma = 300 * 1024 * 1024
        
        #initializing a list of 1000 files we want to transfer
        self.files = deque(maxlen=1000)
        for i in range(1000):
            file_size = int(math.fabs(self.file_size_mean + np.random.standard_normal() * self.file_size_sigma))
            #print("This is the file_size ", file_size)
            self.files.append(file_size)
            
        self.transfers = deque(maxlen=1000)
            
        #print("This is self.transfers: ", self.transfers)

        #  key: int, start: int, stop:int,  size: int [bytes], transfered: int[bytes]
        # self.transfers = deque(maxlen=2000)
        self.current_base_rate = int(self.max_link_rate * 0.5 * np.random.ranf())
        self.tstep = 0
        self.viewer = None
        self.h_base = deque(maxlen=600)
        self.h_added = deque(maxlen=600)
        self.dc_free = 0
        self.dc_used = 0
        self.seed()

        # obesrvation space reports only on files transfered: rate and how many steps ago it started.
        self.observation_space = spaces.Box(
            # low=np.array([0.0, 0, 0]),
            # high=np.array([np.finfo(np.float32).max, np.iinfo(np.int32).max, np.iinfo(np.int32).max])
            low=np.array([0.0]),
            high=np.array([1.5])
            #0 to 1.5 are the percentage threshold. occupency
        )
        self.action_space = spaces.Discrete(4)
        #self.action_space = spaces.Tuple([spaces.Discrete(4), spaces.Box(low=0, high=1, shape=(1,))])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward_function(self, x):
        return -21.22 * x * x * x * x + 33.77 * x * x * x - 15.73 * x * x + 3.306 * x + 0.002029
        #this function is written this way so that the closer our agent approaches the threshold, the more reward it will get, but once it exceeds this threshold, gets negative reward.
        #We can try different reward functions, e.g. one that is linear and then drops down (would also work)
        #Reward funtion below: for each file you get a point, every point past the threshold, you get negative points.
        #if x < 1:
        #    return x
        #else:
        #    return -x + 1
        #Reward function below: same as the one on top, except place more emphasis on everything
        #if x < 1:
        #    return 2*x
        #else:
        #    return -2*x + 2

    def step(self, action):
        
        # add transfers if asked for
        #print("This is action[0]: ", action[0])
        for i in range(action):
            file_size = int(math.fabs(self.file_size_mean + np.random.standard_normal() * self.file_size_sigma))
            #length_files = len(self.files)
            #file_size = self.files[int(round(action[1][0]*length_files))]
            
            #Every time we preform an action (0, 1, 2, or 3) one or more file sizes are generated. E.g. if our action is 2, then 2 file sizes (e.g. 100 bits, 1500 bits) are generated.
            self.transfers.append([self.tstep, 0, file_size, 0])
            #  key: int, start: int, stop:int,  size: int [bytes], transfered: int[bytes]
            #tstep = time step

        # find current base rate
        self.current_base_rate += int(np.random.standard_normal() * 8 * 1024 * 1024)
        #fluctuates the base rate: this generates background. So not every step is the same size.
        if self.current_base_rate > self.base_rate_max:
            self.current_base_rate = self.base_rate_max
        if self.current_base_rate < self.base_rate_min:
            self.current_base_rate = self.base_rate_min

        # find used rate if all the ongoing transfers would be at maximal rate
        active_transfers = 0
        for t in self.transfers:
            # print(t)
            if self.tstep < self.handshake_duration + t[0] or t[1] > 0:
            #Finds how many files are in the progress of being transfered
        #        print("continue ...", self.tstep, t)
                continue
            active_transfers += 1
        #    print("active_transfers = ", active_transfers)

        self.max_rate = self.max_rate_per_file * active_transfers

        # find free bandwidth
        self.max_free_bandwidth = self.max_link_rate - self.current_base_rate
        
        #dc = duty cycle
        self.dc_free += self.max_free_bandwidth / 1024
        self.dc_used += min(self.max_free_bandwidth, self.max_rate) / 1024

        reward = self.reward_function(self.max_rate / self.max_free_bandwidth)
        print("reward is = ", reward)

        episode_over = False
        if (self.max_rate + self.current_base_rate) > 1.1 * self.max_link_rate or self.tstep >= 1400:
            episode_over = True

        current_rate_per_file = 0
        if active_transfers > 0:
            current_rate_per_file = min(math.floor(self.max_free_bandwidth / active_transfers), self.max_rate_per_file)
    
        # LSFT - last started finished transfer
        time_of_LSFT = 0  # how long ago that transfer ended
        rate_of_LSFT = 0
        size_of_LSFT = 0
        finished = 0
        # transfer [start_time, end_time, size, transfered_till_now]
        for t in self.transfers:
            if self.tstep < self.handshake_duration + t[0]:  # still in handshake phase
                continue
            if t[1] == 0:  # increase transfered size for unfinished transfers
                t[3] += current_rate_per_file

            if t[3] >= t[2] and t[1] == 0:  # if some finished in this timestep
                #t[3] is number of bytes transfered and t[2] is the file size in byte. So, when t[3] >= t[2], the file is finished being transfered.
                t[1] = self.tstep
                # t[1] is the timestep that the files transfer is complete.

            if t[3] >= t[2]:  # all finished
                finished += 1  # this is just for info
                if t[0] > time_of_LSFT:  # last started from all finished i.e. the # of timesteps from the previously finished transfer
                    rate_of_LSFT = t[2] / (t[1] - t[0] - self.handshake_duration + 1)
                    size_of_LSFT = t[2]
                    time_of_LSFT = self.tstep - t[1] #This is a relative time, measuring the time from your current time step to that of the previously finished transfer's end timestep.

        size_of_LSFT = 0
        rate_of_LSFT = 0
        time_of_LSFT = self.max_free_bandwidth / self.max_link_rate  # hack

        # observation = (rate_of_LSFT, size_of_LSFT, time_of_LSFT)
        
        
        observation = ((self.max_rate + self.current_base_rate) / self.max_link_rate)
        self.tstep += 1

        self.h_base.append(self.current_base_rate)
        self.h_added.append(self.max_rate + self.current_base_rate)

        return observation, reward, episode_over, {
            "finished transfers": finished,
            "duty cycle": self.dc_used / self.dc_free,
            "active transfers": active_transfers,
            "base rate [%]": int(self.current_base_rate / self.max_link_rate * 10000) / 100
        }

    def reset(self):
        self.tstep = 0
        self.transfers.clear()
        self.dc_free = 0
        self.dc_used = 0
        return np.array([0.5])
        # return np.array((0, 0, 0))

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 640
        screen_height = 480

        scale = np.max(self.h_added) / 440

        bdata = []  # (screen_width - 20, 20)]  # first point in lower right corner
        #This is amount of data that already exists on the link 
        y = list(reversed(self.h_base))
        for j, i in enumerate(y):
            bdata.append((screen_width - 20 - j, 20 + int(i / scale)))
        # bdata.append((screen_width - 20 - len(y), 20))

        adata = []  # (screen_width - 20, 20)]
        #This is amount of data that we are sending through the link 
        y = list(reversed(self.h_added))
        for j, i in enumerate(y):
            adata.append((screen_width - 20 - j, 20 + int(i / scale)))
        # adata.append((screen_width - 20 - len(y), 20))
        adata = adata[:self.tstep]
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
        #     l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        #     axleoffset = cartheight / 4.0
        #     cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     self.carttrans = rendering.Transform()
        #     cart.add_attr(self.carttrans)
        #     self.viewer.add_geom(cart)
        #     self.poletrans = rendering.Transform(translation=(0, axleoffset))
        #     pole.add_attr(self.poletrans)
        #     pole.add_attr(self.carttrans)
        #     self.axle = rendering.make_circle(polewidth / 2)
        #     self.axle.add_attr(self.poletrans)
        #     self.axle.add_attr(self.carttrans)
            self.xaxis = rendering.Line((20, 20), (screen_width - 20, 20))
            self.xaxis.set_color(0, 0, 0)
            self.yaxis = rendering.Line((20, 20), (20, screen_height - 20))
            self.yaxis.set_color(0, 0, 0)
            self.viewer.add_geom(self.xaxis)
            self.viewer.add_geom(self.yaxis)

        adde = rendering.PolyLine(adata, False)
        adde.set_color(.1, .6, .8)
        self.viewer.add_onetime(adde)

        base = rendering.PolyLine(bdata, False)
        base.set_color(.8, .6, .4)
        self.viewer.add_onetime(base)

        max_line = self.max_link_rate / scale
        ml = rendering.Line((20, max_line + 20), (screen_width - 20, max_line + 20))
        ml.set_color(0.1, 0.9, .1)
        self.viewer.add_onetime(ml)

        # if self.state is None:
        # return None

        # x = self.state
        # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
