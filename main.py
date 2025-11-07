import torch
import os
from data_loader import load_dataset
from environment import PowerFlowEnvironment
from agent import PPOAgent
from trainer import train_ppo
from config import DATA_PATHS, ENV_CONFIG


def main():
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")

    os.makedirs('./output', exist_ok=True)

    dataset = load_dataset()

    env = PowerFlowEnvironment(
        num_generators=ENV_CONFIG['num_generators'],
        num_loads=ENV_CONFIG['num_loads'],
        load_converge_model_path=DATA_PATHS['discriminator_model'],
        dataset=dataset
    )

    input_dims = (ENV_CONFIG['num_generators'] * 2) + (ENV_CONFIG['num_loads'] * 2)
    agent = PPOAgent(input_dims, ENV_CONFIG['num_generators'])

    training_results = train_ppo(env, agent)

    print(f"\nBest Reward: {training_results['best_reward']}")


if __name__ == "__main__":
    main()