import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # do not modify these
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

        # modify these
        self.storage = []  # a data structure of your choice (D in the Algorithm 2)
        # A neural network MLP model which can be used as Q
        self.network = MLPRegression(input_dim= 4, output_dim=2, learning_rate=1e-3)
        # network2 has identical structure to network1, network2 is the Q_f
        self.network2 = MLPRegression(input_dim=4, output_dim=2, learning_rate=1e-3)
        # initialise Q_f's parameter by Q's, here is an example
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        self.epsilon = 1.0  # probability ε in Algorithm 2
        self.n = 32  # the number of samples you'd want to draw from the storage each time
        self.discount_factor = 0.99  # γ in Algorithm 2

        # store previous info
        self.previous_state = None
        self.previous_action = None


        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)
    
    #helper for rule-based agent: Not main DQN(just a fallback baseline)
    def get_next_pipe(self, state):
        bird_x = state["bird_x"]
        bird_width = state["bird_width"]

        candidates = []
        for pipe in state["pipes"]:
            if pipe["x"] + pipe["width"] >= bird_x:
                candidates.append(pipe)

        if len(candidates) == 0:
            return None

        return min(candidates, key=lambda p: p["x"])

    def choose_action(self, state: dict, action_table: dict):
        built_state = self.build_state(state)

        if not hasattr(self, "printed_built_state"):
            print("BUILT STATE:", built_state)
            self.printed_built_state = True
        #temporary rule-based action for now
        bird_center = state["bird_y"] + state["bird_height"] / 2

        next_pipe = self.get_next_pipe(state)

        if next_pipe is None:
            target_y = state["screen_height"] * 0.5
        else:
            target_y = (next_pipe["top"] + next_pipe["bottom"]) / 2

        if bird_center > target_y:
            return action_table["jump"]
        else:
            return action_table["do_nothing"]
        
        
        print(built_state)
        #return a_t
    # New Method: critical to agent learning
    # DQN state building
    def build_state(self, state):

        bird_y = state["bird_y"]
        bird_velocity = state["bird_velocity"]

        next_pipe = self.get_next_pipe(state)

        if next_pipe is None:
            pipe_distance_x = state["screen_width"]
            gap_center = state["screen_height"] / 2
        else:
            pipe_distance_x = next_pipe["x"] - state["bird_x"]
            gap_center = (next_pipe["top"] + next_pipe["bottom"]) / 2

        bird_center = state["bird_y"] + state["bird_height"] / 2

        vertical_distance = bird_center - gap_center

        # normalize values(better for training)
        return np.array([
            bird_y / state["screen_height"],
            bird_velocity / 10,
            pipe_distance_x / state["screen_width"],
            vertical_distance / state["screen_height"]
    ])

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        """
        This function should be called to notify the agent of the post-action observation.
        Args:
            state: post-action state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            None
        """
        # following pseudocode to implement this function
        pass

    def save_model(self, path: str = 'my_model.ckpt'):
        """
        Save the MLP model. Unless you decide to implement the MLP model yourself, do not modify this function.

        Args:
            path: the full path to save the model weights, ending with the file name and extension

        Returns:

        """
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        """
        Load the MLP model weights.  Unless you decide to implement the MLP model yourself, do not modify this function.
        Args:
            path: the full path to load the model weights, ending with the file name and extension

        Returns:

        """
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        """
        Update one MLP model's model parameter by the parameter of another MLP model.
        Args:
            net_to_update: the MLP to be updated
            net_as_source: the MLP to supply the model parameters

        Returns:
            None
        """
        net_to_update.load_state_dict(net_as_source.state_dict())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)

    args = parser.parse_args()

    # bare-bone code to train your agent (you may extend this part as well, we won't run your agent training code)
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=True, level=args.level, game_length=10)
    agent = MyAgent(show_screen=True)
    episodes = 10000
    for episode in range(episodes):
        env.play(player=agent)

        # env.score has the score value from the last play
        # env.mileage has the mileage value from the last play
        print(env.score)
        print(env.mileage)
 
        # store the best model based on your judgement
        agent.save_model(path='my_model.ckpt')

        # you'd want to clear the memory after one or a few episodes
        ...

        # you'd want to update the fixed Q-target network (Q_f) with Q's model parameter after one or a few episodes
        ...

    # the below resembles how we evaluate your agent
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')

    episodes = 10
    scores = list()
    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)

    print(np.max(scores))
    print(np.mean(scores))
