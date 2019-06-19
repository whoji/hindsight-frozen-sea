# to create a gym-env like env for frozen sea
# frozen sea is like frozen lake, but YUUUUGE !
# ref: https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai
# ref: https://github.com/openai/gym/blob/master/docs/creating-environments.md
# ref: https://github.com/alibaba/gym-starcraft
# ref: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py


import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

VISION = 0

class FrozenSeaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, w, h, p, genesis_method, seed):
        '''
        genesis_method: {'uniform', 'cluster'}
        '''
        self.w = w
        self.h = h
        self.p = p
        self.seed = seed
        self.world_status = 0
        self.grid = self._genesis(w, h, p, genesis_method, seed)
        self.loc = (0, 0)
        self.a = None
        self.b = None
        self.feasible = None

        #self.reset() # it seems other gym env does not this

    def _genesis(self, w, h, p, method, seed):
        np.random.seed(seed)
        grid = np.zeros([h,w])
        p = [pi/sum(p) for pi in p]
        if method == 'uniform':
            grid = np.random.choice(len(p),size=(h, w),replace=True,p=p)
        elif method == 'cluster':
            raise NotImplementedError
        else:
            raise Exception
        np.random.seed(None)
        return grid

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        self.status = self.env.step()
        reward = self._get_reward()
        ob = self.env.getState()
        episode_over = self.status != hfo_py.IN_GAME
        return ob, reward, episode_over, {}

    def reset(self):
        while 1:
            a_r = np.random.choice(self.h)
            a_c = np.random.choice(self.w)
            if self.grid[a_r][a_c] == 0:
                self.a = (a_r, a_c)
                break
        while 1:
            b_r = np.random.choice(self.h)
            b_c = np.random.choice(self.w)
            if self.grid[b_r][b_c] == 0 and (a_r, a_c) != (b_r, b_c):
                self.b = (b_r, b_c)
                break
        s0 = [a_r, a_c]
        if VISION == 1:
            s0 = s0 + self.get_surround(s0)
        return s0

    def render(self, mode='human', close=False):
        for i in range(self.h):
            if i == self.loc[0]:
                temp_row = [str(b) for b in self.grid[i]]
                #temp_row[self.loc[1]] = str(temp_row[self.loc[1]]) + "*"
                temp_row[self.loc[1]] = '*'
                temp_str = " ".join(temp_row)
                print("["+temp_str+"]")
            else:
                print(self.grid[i])
        print("")

    def take_action(self, action):
        # update the loc
        new_loc, loc_valid = self._get_new_loc(action)
        if  loc_valid:
            self.loc = new_loc

        # calculate the reward
        r = self.get_reward(new_loc)
        r = -1

        # determine the game status
        terminal = False
        if new_loc == self.b:
            r = 100
            terminal = True
        if self.grid[new_loc[0]][new_loc[1]] == 2:
            r = -100
            terminal = True

        s_new = [new_loc[0], new_loc[1]]
        if VISION == 1:
            s_new = s_new + self.get_surround(s_new)

        return s_new, r, terminal, None

    def get_reward(self, new_loc):
        if new_loc == self.b:
            # goal !!!
            return 100
        elif self.grid[new_loc[0]][new_loc[1]] == 2:
            return -100
        else:
            return -1

    def action_space(self):
        return [0, 1, 2, 3, 4] # idle/up/down/left/right

    def observation_space(self):
        import itertools
        h_coord = list(range(self.h))
        w_coord = list(range(self.w))
        return list(itertools.product(h_coord,w_coord))

    def get_obs_size(self):
        s_size = 2
        if VISION == 1:
            s_size += 4
        return s_size

    def get_n_actions(self):
        return len(self.action_space())

    def _get_new_loc(self, action):
        new_loc = list(self.loc)
        if action == 0:
            pass
        elif action == 1:
            if new_loc[0] == 0:
                pass
            else:
                new_loc[0] -= 1
        elif action == 2:
            if new_loc[0] >= self.h-1:
                pass
            else:
                new_loc[0] += 1
        elif action == 3:
            new_loc[1] -= 1 if new_loc[1] > 0 else 0
        elif action == 4:
            new_loc[1] += 1 if new_loc[1] < self.h - 1 else 0
        else:
            assert False

        if not self._if_position_valid(new_loc):
            return self.loc, 0
        else:
            return tuple(new_loc), 1

    def _if_position_valid(self, new_loc):
        if self.grid[new_loc[0]][new_loc[1]] == 0:
            return 1
        elif self.grid[new_loc[0]][new_loc[1]] == 1:
            # impassable
            return 0
        elif self.grid[new_loc[0]][new_loc[1]] == 2:
            # die
            return 1

    def get_surround(self, s):
        ret = []
        try:
            ret.append(self.grid[s[0]][s[1]])
        except:
            ret.append(-1)

        try:
            ret.append(self.grid[s[0]-1][s[1]])
        except:
            ret.append(-1)

        try:
            ret.append(self.grid[s[0]+1][s[1]])
        except:
            ret.append(-1)

        try:
            ret.append(self.grid[s[0]][s[1]-1])
        except:
            ret.append(-1)

        try:
            ret.append(self.grid[s[0]][s[1]+1])
        except:
            ret.append(-1)

        return ret

if __name__ == '__main__':
    env = FrozenSeaEnv(30, 20, (80,10,5), 'uniform' ,12345)
    env.render()
    env.take_action(4);env.render()
    env.take_action(4);env.render()
    # import pdb; pdb.set_trace()