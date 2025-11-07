import torch
import numpy as np

from environment import ImprovedPhaseManager
from utils import RewardTracker, TrainingDataRecorder
from config import TRAIN_CONFIG


def train_ppo(env, agent, max_epochs=TRAIN_CONFIG['max_epochs'],
              max_steps_per_epoch=TRAIN_CONFIG['max_steps_per_epoch']):
    actor_losses = []
    critic_losses = []
    rewards_history = []

    recorder = TrainingDataRecorder(output_dir="output")
    total_phase_transitions = 0
    phase_transition_epochs = []

    best_reward = float('-inf')
    patience = 50
    patience_counter = 0

    for epoch in range(max_epochs):
        state = env.reset()
        total_reward = 0
        epoch_steps = 0
        phase_switch_step = None
        done = False

        env.phase_manager = ImprovedPhaseManager()
        epoch_phase_history = []
        reward_tracker = RewardTracker()

        for step in range(max_steps_per_epoch):
            if done:
                break

            action, prob, val = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            current_phase = env.phase_manager.phase
            epoch_phase_history.append(current_phase)

            if phase_switch_step is None and current_phase == 2:
                phase_switch_step = step
                total_phase_transitions += 1
                print(f"Epoch {epoch}, Step {step}: Phase 1 -> Phase 2 transition")

            agent.update_phase(reward)
            reward_tracker.add_reward(reward)
            agent.store_transition(state, action, prob, val, reward, done, current_phase)

            state = next_state
            total_reward += reward
            epoch_steps += 1

        if phase_switch_step is not None:
            phase_transition_epochs.append({
                'epoch': epoch,
                'transition_step': phase_switch_step,
                'total_steps': epoch_steps
            })

        reward_tracker.plot_rewards(title="PPO Training Epoch Rewards", epoch=epoch)
        actor_loss, critic_loss = agent.learn()

        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        avg_reward = total_reward / epoch_steps
        rewards_history.append(avg_reward)

        phase_switch_step = phase_switch_step if phase_switch_step is not None else 0
        recorder.record_epoch_data(epoch, epoch_steps, avg_reward, phase_switch_step)

        final_phase = epoch_phase_history[-1] if epoch_phase_history else 1
        transition_info = f"(Transition at step {phase_switch_step})" if phase_switch_step else ""

        print(f"Epoch {epoch:4d} | Final Phase: {final_phase} | Steps: {epoch_steps:3d} | "
              f"Avg Reward: {avg_reward:8.2f} | Actor Loss: {actor_loss:8.4f} | "
              f"Critic Loss: {critic_loss:8.4f} {transition_info}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            patience_counter = 0
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'best_reward': best_reward
            }, './PPO-best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    summary_file = recorder.save_summary()
    print(f"训练数据已保存到: output")
    print(f"摘要文件: {summary_file}")

    return {
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'rewards': rewards_history,
        'best_reward': best_reward,
        'total_phase_transitions': total_phase_transitions
    }