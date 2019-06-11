# to create a gym-env like env for frozen sea
# frozen sea is like frozen lake, but YUUUUGE !
# ref: https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai
# ref: https://github.com/openai/gym/blob/master/docs/creating-environments.md
# ref: https://github.com/alibaba/gym-starcraft

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, w, h, p, genesis_method, seed):
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
        pass

    def take_action(self, action):
        pass

    def get_reward(self):
        """ Reward is given for XY. """
        if self.status == FOOBAR:
            return 1
        elif self.status == ABC:
            return self.somestate ** 2
        else:
            return 0

    def action_space(self):
        pass

    def observation_space(self):
        pass