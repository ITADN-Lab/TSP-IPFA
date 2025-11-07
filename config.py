import os

# 训练配置
TRAIN_CONFIG = {
    'max_epochs': 500,
    'max_steps_per_epoch': 5000,
    'learning_rate': 1e-2,
    'gamma': 0.99,
    'epsilon': 0.2,
    'batch_size': 128,
    'n_epochs': 4
}

# 环境配置
ENV_CONFIG = {
    'num_generators': 5,
    'num_loads': 20,
    'voltage_limits': (0.95, 1.05),
    'thermal_limit': 100.0
}

# 文件路径配置
DATA_PATHS = {
    'database': "./case30_samples_PQ_thin.db",
    'discriminator_model': "./best_discriminator_二分类_许.pth",
    'discriminator_data': "./best_discriminator_二分类_许_data.pkl"
}

# 可视化配置
PLOT_CONFIG = {
    'font_family': 'Times New Roman',
    'font_size': 21,
    'dpi': 300
}