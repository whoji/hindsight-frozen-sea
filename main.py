import torch
import torch.nn as nn
import numpy as np
from model import PolicyNet
from frozen_sea_env import FrozenSeaEnv

H = 10
W = 10
P = [70, 20, 10]
SEED = 333


def play_one_episode(env, net, max_steps=200):
    s = env.reset()
    Ret = 0.0
    steps = 0
    data = []
    for i in range(max_steps):
        s_v = torch.FloatTensor([s])
        # print(s_v)
        nn_output_v = net(s_v)
        act_probs_v = nn.Softmax(dim=1)(nn_output_v)
        act_probs = act_probs_v.data.numpy()[0]
        a = np.random.choice(len(act_probs), p=act_probs)
        # print(a)
        data.append((s, a))
        s_new, r, terminal, _ = env.take_action(a)
        Ret += r
        steps += 1
        if terminal:
            break
        else:
            s = s_new
    return Ret, steps, data

def test_play(env, net, episodes=100, verbose=True):
    for i in range(episodes):
        Ret, steps, _  = play_one_episode(env, net)
        print("ep %d | %d steps | rwd: %.3f " % (i, steps, Ret) )

def train_CE(env, net, ep_max=100000, cutoff=70, top_perc=0.2):
    pass

def train_PG(env, net):
    pass

def train_GA(env, net):
    pass

def main():
    env = FrozenSeaEnv(H,W,P,'uniform',SEED)
    net = PolicyNet(env.get_obs_size(), env.get_n_actions())
    print(net)
    test_play(env, net, 10)

if __name__ == '__main__':
    main()