"""

"""

import gym
import logging
import pandas as pd

from abc import ABC, abstractmethod
from stable_baselines3 import PPO
from src.envs.fraud_env import FraudEnv

class Model(ABC):
    """
    Abstract class for all models.
    """

    @abstractmethod
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the model.

        Args:
            X_train: the training data
            y_train: the training labels
        Returns:
            None
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str):
        """
        Loads a pre-trained model.

        Args:
            model_path: path to the pre-trained model
        Returns:
            None
        """
        pass

class PPOAgent(Model):
    """
    PPO Agent model for training.
    """

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> PPO:
        """
        Trains the PPO Agent model.

        Args:
            X_train: the training data
            y_train: the training labels
        Returns:
            None
        """
        logging.info("Training PPO Agent model...")

        train_df = X_train.copy()
        train_df["Class"] = y_train.values

        env = FraudEnv(train_df)

        vec_env = gym.vector.SyncVectorEnv([lambda: env])

        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            learning_rate=0.0003,
            batch_size=64,
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0
        )

        num_epochs = 10

        model.learn(total_timesteps=(len(train_df) * num_epochs))
        model.save("models/ppo_agent")

        logging.info("Training completed and model saved.")
        return model
    
    def load_model(self, model_path) -> PPO:
        return PPO.load(model_path)