# to create a gym-env like env for frozen sea
# frozen sea is like frozen lake, but YUUUUGE !
# ref: https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai
# ref: https://github.com/openai/gym/blob/master/docs/creating-environments.md
# ref: https://github.com/alibaba/gym-starcraft

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


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
        self.loc = None
        self.a = None
        self.b = None
        self.feasible = None

    def _genesis(self, w, h, p, method, seed):
        grid = np.zeros([h,w])
        p = [pi/sum(p) for pi in p]
        if method == 'uniform':
            grid = np.random.choice(len(p),size=(h, w),replace=True,p=p)
        elif method == 'cluster':
            raise NotImplementedError
        else:
            raise Exception
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
        pass

    def render(self, mode='human', close=False):
        for i in range(self.h):
            print(self.grid[i])

    def take_action(self, action):
        # update the loc
        new_loc, loc_valid = self._get_new_loc(action)
        if  loc_valid:
            self.loc = new_loc

        # calculate the reward
        r = 0
        if loc_valid:
            r += 1

        # determine the game status
        terminal = False
        if new_loc == self.b:
            r = 100
            terminal = True
        if new_loc.is_bad():
            r = -10
            terminal = True

        return new_loc, r, terminal, _

    def get_reward(self):
        """ Reward is given for XY. """
        if self.status == FOOBAR:
            return 1
        elif self.status == ABC:
            return self.somestate ** 2
        else:
            return 0

    def action_space(self):
        return [0, 1, 2, 3, 4]

    def observation_space(self):
        import itertools
        h_coord = list(range(self.h))
        w_coord = list(range(self.w))
        return list(itertools.product(h_coord,w_coord))


if __name__ == '__main__':
    env = FrozenSeaEnv(30, 20, (80,10,5), 'uniform' ,12345)
    env.render()