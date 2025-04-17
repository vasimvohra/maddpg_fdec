import numpy as np
import torch as T
import torch.nn.functional as F
from Classes.networks import ActorNetwork, CriticNetwork
from Classes.noise import OUActionNoise
from Classes.buffer import ReplayBuffer

class Agent:
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma, 
                 C_fc1_dims, C_fc2_dims, A_fc1_dims, A_fc2_dims, 
                 batch_size, n_agents, agent_name, memory_size):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.number_agents = n_agents
        self.number_actions = n_actions
        self.number_states = input_dims
        self.agent_name = agent_name
        self.memory_size = memory_size
        self.local_critic_loss = []

        # Add reward tracking attributes
        self.reward_window = []
        self.running_reward = 0
        self.best_reward = float('-inf')
        self.window_size = 10

        self.memory = ReplayBuffer(self.memory_size, input_dims, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # Initialize networks with agent_name as agent_label
        self.actor = ActorNetwork(alpha, input_dims, A_fc1_dims, A_fc2_dims, n_agents,
                                n_actions=n_actions, name='actor', agent_label=agent_name)
        self.critic = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, n_agents,
                                n_actions=n_actions, name='critic', agent_label=agent_name)
        self.target_actor = ActorNetwork(alpha, input_dims, A_fc1_dims, A_fc2_dims, n_agents,
                                     n_actions=n_actions, name='target_actor', agent_label=agent_name)
        self.target_critic = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, n_agents,
                                       n_actions=n_actions, name='target_critic', agent_label=agent_name)

        self.update_network_parameters(tau=1)

    def update_rewards(self, reward):
        """Update reward statistics for the agent"""
        self.reward_window.append(reward)
        if len(self.reward_window) > self.window_size:
            self.reward_window.pop(0)
        
        self.running_reward = np.mean(self.reward_window)
        if self.running_reward > self.best_reward:
            self.best_reward = self.running_reward

    def get_metrics(self):
        """Return current reward metrics"""
        return {
            'running_avg': self.running_reward,
            'best_avg': self.best_reward,
            'recent_rewards': self.reward_window
        }

    def remember(self, state, action, reward, state_, done):
        """Store transition in memory and update reward statistics"""
        self.memory.store_transition(state, action, reward, state_, done)
        self.update_rewards(reward)        

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def choose_action(self, state, noise_scale=1.0):
        self.actor.eval()
        state = np.array(state, dtype=np.float32)
        state = T.tensor(state.reshape(1, -1), dtype=T.float32).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        noise = T.tensor(self.noise() * noise_scale, dtype=T.float32).to(self.actor.device)
        mu_prime = mu + noise
        self.actor.train()
        return mu_prime.cpu().detach().numpy()[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        # Use running reward instead of reward_tracker
        current_avg = T.tensor(self.running_reward, dtype=T.float).to(self.actor.device)
        target = rewards - current_avg + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()

        self.actor.train()
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()  
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()