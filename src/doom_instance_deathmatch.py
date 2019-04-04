#
# doom_instance.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
from vizdoom import *
import numpy as np
from doom_instance import DoomInstance


class DoomInstanceDeathmatch(DoomInstance):
    def __init__(self, config, wad, skiprate, visible=False, mode=Mode.PLAYER, actions=None, id=None, args=None, config_wad=None, map_id=None):
        super().__init__(config, wad, skiprate, visible, mode, actions, id, args)

    def step(self, action):
        reset_variables = False
        if self.use_action_set:
            action = self.actions[action]

        if self.game.is_player_dead():
            self.game.respawn_player()
            reset_variables = True

        reward = self.game.make_action(action, self.skiprate)

        episode_finished = self.game.is_episode_finished()
        dead = self.game.is_player_dead()
        finished = episode_finished or dead
        if finished:
            # self.episode_return = self.variables[2]
            self.episode_return = self.game.get_total_reward()

        if finished:
            self.new_episode()
            reset_variables = True

        state = self.get_state()

        if reset_variables and state.game_variables is not None:
            self.variables = state.game_variables

        return state, reward, finished, dead

    def rare_action(self, action, rarenum):
        a = action.item()
        if rarenum[a] == 0:
            rarenum[a] = 1
        actreward = 40*50/rarenum[a]
        # print(actreward)
        return actreward

    def step_normalized(self, action, rarenum):
        state, reward, finished, dead = self.step(action)
        # print(action)
        actreward = self.rare_action(action, rarenum=rarenum)
        # print(actreward)
        state = self.normalize(state)

        if state.variables is not None:
            diff = state.variables - self.variables
            # print(state.variables)
            # print(diff,123)
            diff[0] *= 1000
            diff[2] /= 5
            reward += diff.sum()
            self.variables = state.variables.copy()
        # print(actreward)
        reward += actreward
        # print(reward)
        # print(reward)

        return state, reward, finished
