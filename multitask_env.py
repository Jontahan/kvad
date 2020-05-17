import random

class MultitaskEnvironment:
    def __init__(self, env_list=[]):
        self.env_scores = {}
        for env in env_list:
            self.env_scores[env.seed] = []
        self.env_list = env_list
        self.current_env = env_list[random.randint(0, len(env_list) - 1)]
        self.action_space = self.current_env.action_space
        self.duration_counter = 0

    def reset(self):
        self.env_scores[self.current_env.seed].append(self.duration_counter)
        self.current_env = self.env_list[random.randint(0, len(self.env_list) - 1)]
        self.duration_counter = 0
        return self.current_env.reset()

    def step(self, action):
        self.duration_counter += 1
        return self.current_env.step(action)