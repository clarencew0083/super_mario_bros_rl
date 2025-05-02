import os
import torch
import pickle
import numpy as np

def save_checkpoint(agent, scores, eps_history, episode, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(agent.Q_eval.state_dict(), f"{checkpoint_dir}/Q_eval_ep{episode}.pth")
    torch.save(agent.Q_next.state_dict(), f"{checkpoint_dir}/Q_next_ep{episode}.pth")

    # Save epsilon, training steps, and metrics
    with open(f"{checkpoint_dir}/metrics_ep{episode}.pkl", 'wb') as f:
        pickle.dump({
            'scores': scores,
            'eps_history': eps_history,
            'epsilon': agent.epsilon,
            'training_steps': agent.training_steps
        }, f)

    # Save replay buffer
    with open(f"{checkpoint_dir}/replay_buffer_ep{episode}.pkl", 'wb') as f:
        pickle.dump(agent.memory, f)

def save_best_model(agent, scores, eps_history, episode, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(agent.Q_eval.state_dict(), f"{checkpoint_dir}/best_Q_eval.pth")
    torch.save(agent.Q_next.state_dict(), f"{checkpoint_dir}/best_Q_next.pth")

    with open(f"{checkpoint_dir}/best_metrics.pkl", 'wb') as f:
        pickle.dump({
            'scores': scores,
            'eps_history': eps_history,
            'best_episode': episode,
            'epsilon': agent.epsilon,
            'training_steps': agent.training_steps
        }, f)

    with open(f"{checkpoint_dir}/best_replay_buffer.pkl", 'wb') as f:
        pickle.dump(agent.memory, f)

def run_episode(episodes, env, agent, scores, eps_history, batch_size):
    best_avg_score = float('-inf')
    train_freq = 4     # how often to train (in steps)
    target_update_freq = 2000 # how often to update target net (in steps)
    total_steps = 0

    for i in range(episodes):
        state, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            score += reward  # You can still track raw game score

            agent.memorize(state, action, reward, next_state, done)
            state = next_state

            total_steps += 1

            # Train every `train_freq` steps
            if total_steps % train_freq == 0:
                agent.train(batch_size)  # for PER

            # Update target network every `target_update_freq` steps
            if total_steps % target_update_freq == 0:
                agent.update_target_network()


        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-10:])
        print(f"Episode: {i}, Avg Score: {avg_score:.3f}, Epsilon: {agent.epsilon:.3f}")

        # Save best model only
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            save_best_model(agent, scores, eps_history, i)
            print(f"New best model saved at episode {i} with avg score {avg_score:.3f}")

        # Save a checkpoint every 50 episodes
        if (i + 1) % 50 == 0:
            save_checkpoint(agent, scores, eps_history, i + 1)
            
        agent.update_epsilon()  #  Decay epsilon once per episode