import numpy as np
import torch as T
import time
import os
import scipy.io
import Classes.Environment_Platoon as ENV
from ddpg_torch import Agent

def get_state(env, idx, size_platoon):
    """
    Get state information for a specific agent
    
    Args:
        env: Environment object
        idx: Agent index
        size_platoon: Size of each platoon
    """
    V2I_abs = (env.V2I_channels_abs[idx * size_platoon] - 60) / 60.0
    V2V_abs = (env.V2V_channels_abs[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1))] - 60)/60.0
    V2I_fast = (env.V2I_channels_with_fastfading[idx * size_platoon, :] - env.V2I_channels_abs[idx * size_platoon] + 10) / 35
    V2V_fast = (env.V2V_channels_with_fastfading[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1)), :] - 
                env.V2V_channels_abs[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1))].reshape(size_platoon - 1, 1) + 10) / 35
    Interference = (-env.Interference_all[idx] - 60) / 60
    AoI_levels = env.AoI[idx] / (int(env.time_slow / env.time_fast))
    V2V_load_remaining = np.asarray([env.V2V_demand[idx] / env.V2V_demand_size])
    
    return np.concatenate((np.reshape(V2I_abs, -1), np.reshape(V2I_fast, -1), np.reshape(V2V_abs, -1),
                          np.reshape(V2V_fast, -1), np.reshape(Interference, -1), np.reshape(AoI_levels, -1), 
                          V2V_load_remaining), axis=0)
def main():
    # Environment parameters
    n_veh = 20
    size_platoon = 4
    n_platoon = n_veh // size_platoon
    n_RB = 3
    n_input = 19
    # Road parameters
    up_lanes = [0.875, 2.625, 125.875, 127.625, 250.875, 252.625]
    down_lanes = [122.375, 124.125, 247.375, 249.125, 372.375, 374.125]
    left_lanes = [0.875, 2.625, 217.375, 219.125, 433.875, 435.625]
    right_lanes = [213.875, 215.625, 430.375, 432.125, 646.875, 648.625]
    width = 1000
    height = 500
    
    # V2V parameters
    V2V_size = 1060
    V2I_min = 1
    Gap = 0.5
    bandwidth = int(1e6)
    
    # Training parameters
    n_episode = 2000
    n_step_per_episode = 100
    
    # Agent parameters
    alpha = 0.0001
    beta = 0.001
    tau = 0.001
    gamma = 0.95
    batch_size = 64
    memory_size = 1000000
    
    # Network dimensions
    C_fc1_dims = 1024
    C_fc2_dims = 512
    A_fc1_dims = 1024
    A_fc2_dims = 512

    # Initialize environment
    env = ENV.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, size_platoon, n_RB,
                    V2I_min, bandwidth, V2V_size, Gap)
    env.new_random_game()

    # Initialize agents
    agents = []
    for i in range(n_platoon):
        agent = Agent(

            alpha=0.0001,         # Actor learning rate
            beta=0.001,          # Critic learning rate
            input_dims=19,       # State dimension
            tau=0.005,          # Target network update rate
            n_actions=3,         # Number of actions
            gamma=0.99,         # Discount factor
            C_fc1_dims=1024,     # Critic layers
            C_fc2_dims=512,
            
            A_fc1_dims=1024,     # Actor layers
            A_fc2_dims=512,
            batch_size=64,       # Batch size
            n_agents=n_platoon,  # Number of agents
            agent_name=f"agent_{i}",  # Add unique agent name for each agent
            memory_size=100000   # Memory buffer size
    )
        agents.append(agent)

    # Initialize tracking arrays
    record_reward = np.zeros(n_episode)
    AoI_total = np.zeros([n_platoon, n_episode])
    AoI_evolution = np.zeros([n_platoon, 100, n_step_per_episode])
    Demand_total = np.zeros([n_platoon, 100, n_step_per_episode])
    V2I_total = np.zeros([n_platoon, 100, n_step_per_episode])
    V2V_total = np.zeros([n_platoon, 100, n_step_per_episode])
    power_total = np.zeros([n_platoon, 100, n_step_per_episode])

    # Training loop
    for i_episode in range(n_episode):
        print("---------------------------------------")
        record_reward_episode = np.zeros([n_step_per_episode], dtype=np.float16)
        record_AoI = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)

        # Environment reset
        env.V2V_demand = env.V2V_demand_size * np.ones(n_platoon, dtype=np.float16)
        env.individual_time_limit = env.time_slow * np.ones(n_platoon, dtype=np.float16)
        env.active_links = np.ones((n_platoon), dtype='bool')
        
        if i_episode == 0:
            env.AoI = np.ones(n_platoon) * 100

        # Renew environment periodically
        if i_episode % 20 == 0:
            env.renew_positions()
            env.renew_channel(n_veh, size_platoon)
            env.renew_channels_fastfading()

        # Get initial states
        states = [get_state(env, i,size_platoon) for i in range(n_platoon)]

        for i_step in range(n_step_per_episode):
            # Get actions
            actions = []
            noise_scale = max(0.1, 1.0 - i_episode/200)
            
            for i, agent in enumerate(agents):
                action = agent.choose_action(states[i], noise_scale)
                action = np.clip(action, -0.999, 0.999)
                actions.append(action)

            # Process actions
            action_all_training = np.zeros([n_platoon, 3], dtype=np.float32)
            for i in range(n_platoon):
                action_all_training[i, 0] = ((actions[i][0] + 1) / 2) * n_RB
                action_all_training[i, 1] = ((actions[i][1] + 1) / 2) * 2
                action_all_training[i, 2] = ((actions[i][2] + 1) / 2) * 30

            # Environment step
            result = env.act_for_training(action_all_training)
            individual_rewards, global_reward, platoon_AoI, C_rate, V_rate, Demand_R = result

            # Record step results
            record_reward_episode[i_step] = np.mean(global_reward)
            record_AoI[:, i_step] = platoon_AoI

            # Update environment
            env.renew_channels_fastfading()
            env.Compute_Interference(action_all_training)

            # Get next states
            next_states = [get_state(env, i,size_platoon) for i in range(n_platoon)]
            
            # Store experience and learn
            done = (i_step == n_step_per_episode - 1)
            for i, agent in enumerate(agents):
                agent.remember(states[i], actions[i], individual_rewards[i], next_states[i], done)
                if agent.memory.mem_cntr > agent.batch_size:
                    agent.learn()

            states = next_states

            # Update tracking arrays
            for i in range(n_platoon):
                AoI_evolution[i, i_episode % 100, i_step] = platoon_AoI[i]
                Demand_total[i, i_episode % 100, i_step] = Demand_R[i]  # Demand_R is now indexed
                V2I_total[i, i_episode % 100, i_step] = C_rate[i]
                V2V_total[i, i_episode % 100, i_step] = V_rate[i]
                power_total[i, i_episode % 100, i_step] = action_all_training[i, 2]

        # Episode summary
        episode_reward = np.mean(record_reward_episode)
        record_reward[i_episode] = episode_reward
        AoI_total[:, i_episode] = env.AoI

        # Print progress
        if i_episode % 10 == 0:
            print(f"\nEpisode {i_episode}")
            print(f"Episode Reward: {episode_reward:.4f}")
            print(f"Average AoI: {np.mean(AoI_total[:, i_episode]):.4f}")
            print(f"Average V2I Rate: {np.mean(V2I_total[:, i_episode % 100]):.4f}")
            print(f"Average V2V Rate: {np.mean(V2V_total[:, i_episode % 100]):.4f}")
            
            for i, agent in enumerate(agents):
                metrics = agent.get_metrics()
                print(f"\nAgent {i} metrics:")
                print(f"Running Average (last 10): {metrics['running_avg']:.4f}")
                print(f"Best Average: {metrics['best_avg']:.4f}")

        # Save models periodically
        if i_episode % 500 == 0:
            for agent in agents:
                agent.save_models()

    # Save final results
    print('Training Done. Saving models...')
    
    # Save training history
    scipy.io.savemat('train_history.mat', {
        'record_reward': record_reward,
        'AoI_total': AoI_total,
        'AoI_evolution': AoI_evolution,
        'Demand_total': Demand_total,
        'V2I_total': V2I_total,
        'V2V_total': V2V_total,
        'power_total': power_total
    })

if __name__ == "__main__":
    main()