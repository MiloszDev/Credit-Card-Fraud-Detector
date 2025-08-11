import gym
import pandas as pd
import numpy as np

class FraudEnv(gym.Env):
    """
    Environment for fraud detection tasks.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initializes the FraudEnv with the provided data.

        Args:
            data: DataFrame containing the fraud detection data.
        """
        super(FraudEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.current_index = 0
        self.max_steps = len(data) - 1
        self.action_space = gym.spaces.Discrete(2)  # 0 = Not Fraud, 1 = Fraud

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(data.shape[1] - 1,), dtype=np.float32
        )

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
            Initial observation of the environment.
        """
        info = {}
        self.current_step = 0
        self.last_action = None

        features = self.data.drop(columns=["Class"])
        return features.iloc[self.current_step].values.astype(np.float32), info

    def step(self, action: int):
        """
        Executes one step in the environment.

        Args:
            action: Action taken by the agent (0 or 1).

        Returns:
            Tuple containing:
                - observation: Next observation of the environment.
                - reward: Reward received after taking the action.
                - done: Boolean indicating if the episode has ended.
                - info: Additional information.
        """
        reward = 0.0
        label = self.data.loc[self.current_step, "Class"]

        if action == label:
            reward = 5.0 if label == 1 else 1.0
        else:
            reward = -10.0 if label == 1 else -1.0

        self.last_action = action
        self.current_step += 1
        
        terminated = self.current_index >= len(self.data)
        truncated = self.current_step >= self.max_steps

        features = self.data.drop(columns=["Class"])

        if not (terminated or truncated):
            obs = features.iloc[self.current_step].values.astype(np.float32)
        else:
            obs = np.zeros(features.shape[1], dtype=np.float32)

        return obs, reward, terminated, truncated, {}

    def render(self, mode='human'):
        """
        Renders the current step of the environment.

        Args:
            mode (str): The mode to render with. Currently supports 'human'.
        """
        if self.current_step == 0:
            print("Environment reset. Ready to start.")
        else:
            label = self.data.loc[self.current_step - 1, "Class"]
            action = self.last_action
            print(f"Step: {self.current_step} | True Label: {label} | Agent Action: {action}")

