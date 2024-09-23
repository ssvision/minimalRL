import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
from tqdm import tqdm

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f"device is {self.device}")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class PolicyGradienAgent():
    def __init__(self, lr, input_dims,  gamma=0.99, n_actions=4):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

    def choose_action(self, observation):
        state = T.tensor(observation).to(self.policy.device)
        # print(f"State is : {state}")
        # print(f"Shape of state is : {state.shape}")
        probabilities = F.softmax(self.policy(state))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()
    
    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        #G_t = R_t+1 + gamma * R_t+2 * gamma^2 * R_t+3 .....

        G = np.zeros_like(self.reward_memory)

        for t in range(len(self.reward_memory)):

            G_sum = 0
            discount = 1

            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount*= self.gamma
            G[t] = G_sum
        G = T.tensor(G).to(self.policy.device)

        loss = 0

        for g,logprob in zip(G, self.action_memory):
            loss += -g * logprob
        
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []



    


if __name__ == "__main__":


    env = gym.make("LunarLander-v2")
    n_episodes = 3000

    agent = PolicyGradienAgent(lr = 0.0005,  input_dims=8, gamma=0.99, n_actions=4)


    returns = []

    for i in (range(n_episodes)):

        terminated = False
        truncated = False
        _current_state = env.reset()
        current_state = _current_state[0]
        episode_return = 0
        
        # print(f" current_state is : {(current_state)}")
        # print(f"Dtype of current_state is : {type(current_state)}")

        while not terminated or not truncated:

            action = agent.choose_action(current_state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # print(f"next_state is : {(next_state)}")
            # print(f"type of next_state is : {(type(next_state))}")

            episode_return+=reward 
            agent.store_rewards(reward)
            current_state = next_state
        
        agent.learn()
        returns.append(episode_return)

        avg_return = np.mean([returns[-50:]])
        
        print(f"Episode  {i} ||  Episode return : {episode_return} || Average Return :{avg_return}")














