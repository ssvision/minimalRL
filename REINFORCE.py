import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from tqdm import tqdm
import matplotlib.pyplot as plt


#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

def main():

    #Define your environment here
    env = gym.make('CartPole-v1')

    #Create the Policy Network Object
    pi = Policy()
    score = 0.0


    print_interval = 20
    rewards = []

    #Define the num_episodes here (10k-1M)
    n_eps = 10
    
    
    for n_epi in tqdm(range(n_eps)):
        s, _ = env.reset()
        done = False
        
        while not done: # CartPole-v1 forced to terminates at 500 step.

            prob = pi(torch.from_numpy(s).float()) #Get the action probs here
            # print(f"prob is : {prob}") 
            m = Categorical(prob) #Create a distribution for the actions based on probs
            # print(f"m is : {m}")
            a = m.sample() #Sample an action from the distribution
            # print(f"a is : {a}")

            
            s_prime, r, done, truncated, info = env.step(a.item())
            pi.put_data((r,prob[a]))
            s = s_prime
            score += r
            
        pi.train_net()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            rewards.append(score/print_interval)
            score = 0.0
    env.close()
    plt.plot(rewards)
    plt.show()

    
if __name__ == '__main__':
    main()