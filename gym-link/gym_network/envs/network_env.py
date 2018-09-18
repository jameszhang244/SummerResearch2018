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
from gym_link.envs.link_env import LinkEnv

import numpy as np

class NetworkEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        #max_rate is the amount in gigabytes of info you're sending out.
        self.max_rate = 0
        self.max_link_rate = 10 * 1024 * 1024 * 1024 / 8  # 10 Gigabits - all rates are in B/s
        
        self.max_read_rate = self.max_link_rate * 4 * 2 / 3 # Simulates rate for 4 links
        
        #  key: int, start: int, stop:int,  size: int [bytes], transfered: int[bytes]
        self.network_transfers = deque(maxlen=4)
        self.network_transfers.append(deque(maxlen=2000))
        self.network_transfers.append(deque(maxlen=2000))
        self.network_transfers.append(deque(maxlen=2000))
        self.network_transfers.append(deque(maxlen=2000))
        
        self.current_base_rate = int(self.max_read_rate * 0.5 * np.random.ranf())
        
        self.tstep = 0
        self.viewer = None
        self.h_base = deque(maxlen=600)
        self.h_added = deque(maxlen=600)
        self.links = deque(maxlen=4)
        self.links.append(LinkEnv())
        self.links.append(LinkEnv())
        self.links.append(LinkEnv())
        self.links.append(LinkEnv())
        self.seed()
        
        """
        low = np.array([
                self.links[0].step[0], 
                self.links[1].step[0], 
                self.links[2].step[0], 
                self.links[3].step[0], 
                0.5])
        
        high = low
        """
        
        # obesrvation space reports only on files transfered: rate and how many steps ago it started.
        self.observation_space = spaces.Box(
            # low=np.array([0.0, 0, 0]),
            # high=np.array([np.finfo(np.float32).max, np.iinfo(np.int32).max, np.iinfo(np.int32).max])

            low=0.0,
            high=1.5,
            shape=(1,4)
            # 4 observations for the throughput on each link and the fifth being if these throughputs exceed the read rate.
            #0 to 1.5 are the percentage threshold. occupency
        )
        self.action_space = spaces.Tuple([spaces.Discrete(4), spaces.Discrete(4), spaces.Discrete(4), spaces.Discrete(4)])
        #self.action_space = spaces.MultiDiscrete([4, 4, 4, 4])
        #self.action_space = spaces.Discrete(16)
        
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
        
    def round_int(self, x):
        return int(round(x))       
    
    def get_max_rate(self):
        max0 = self.links[0].max_rate
        print("This is the max0 rate ", max0)
        max1 = self.links[1].max_rate
        print("This is the max1 rate ", max1)
        max2 = self.links[2].max_rate
        print("This is the max2 rate ", max2)
        max3 = self.links[3].max_rate
        print("This is the max3 rate ", max3)
        maxr = max0 + max1 + max2 + max3
        return maxr
      
    
    def get_current_base_rate(self):
        cbr0 = self.links[0].current_base_rate
        print("This is the current max0 base rate ", cbr0)
        cbr1 = self.links[1].current_base_rate
        print("This is the current max1 base rate ", cbr1)
        cbr2 = self.links[2].current_base_rate
        print("This is the current max2 base rate ", cbr2)
        cbr3 = self.links[3].current_base_rate
        print("This is the current max3 base rate ", cbr3)
        cbr = cbr0 + cbr1 + cbr2 + cbr3
        return cbr

    def step(self, action):

        #print("This is action ", action)
        
        self.max_rate = self.get_max_rate()
        #print("This is the max_rate ", self.max_rate)
        #print("This is the link ", self.links)
        
        self.current_base_rate = self.get_current_base_rate()
        #print("This is the current_base_rate ", self.current_base_rate)
        
        self.max_free_network_bandwidth = self.max_read_rate - self.current_base_rate
        #print("This is the max_free_network_bandwidth ", self.max_free_network_bandwidth)
        
        self.h_base.append(self.current_base_rate)
        #print("This is the h_base ", self.h_base)
        
        self.h_added.append(self.max_rate + self.current_base_rate)
        #print("This is the h_added ", self.h_added)

        # add transfers if asked for
        l = np.array(action)
        print("This is l ok ", l)
        list1 = list(enumerate(l))
        print("This is list1 ", list1)

        step_results = []
        obs = []
        rew = []
        #rew = self.reward_function(self.max_rate / self.max_free_network_bandwidth)
        #print("reward prlim is ", rew)
        
        for (idx, act) in list1:
            o, r, e, _ = self.links[idx].step(act)
            step_results.append([o, r, e])
            obs.append(o)
            rew.append(r)
            #print("OK SO THIS IS rew each time: ", rew)
        #print("reward after is ", rew)
            
        epi_over = True
        for i in step_results:
            #obs += i[0]
            #rew += i[1]
            epi_over = epi_over and i[2]
      
        observation = obs
        reward = rew
        #print("THE reward is ", reward)
        episode_over = epi_over
        
        self.tstep += 1
        
        # TO DO:
        #observation = np.array([0.6, 0.6, 0.6, 0.6])
       
        return observation, reward, episode_over, {
            "finished network transfers": True
        }

    def reset(self):
        self.tstep = 0
        s = self.links[0].reset()
        for i in range(1, 4):
            a = self.links[i].reset()
            s = np.concatenate((s, a), axis=0)
        print("network reset: s = ", s)
        return s
        #return np.array([0.5, 0.5, 0.5, 0.5])
        

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

        max_line = self.max_read_rate / scale
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
