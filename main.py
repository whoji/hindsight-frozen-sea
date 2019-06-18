import torch
import torch.nn as nn
import numpy as np
from model import PolicyNet
from frozen_sea_env import FrozenSeaEnv

H = 50
W = 30
P = [70, 20, 10]
SEED = 333

def test_play(env, net, steps=200, verbose=True):
    s = env.reset()
    for i in range(steps):
        s_v = torch.FloatTensor([s])
        print(s_v)
        nn_output_v = net(s_v)
        act_probs_v = nn.Softmax(dim=1)(nn_output_v)
        act_probs = act_probs_v.data.numpy()[0]
        a = np.random.choice(len(act_probs), p=act_probs)
        print(a)

        s_new, r, terminal, _ = env.take_action(a)
        if terminal:
            break
        else:
            s = s_new

def main():
    env = FrozenSeaEnv(H,W,P,'uniform',SEED)
    net = PolicyNet(env.get_obs_size(), env.get_n_actions())
    print(net)
    test_play(env, net)

if __name__ == '__main__':
    main()