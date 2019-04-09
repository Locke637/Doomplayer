#
# ppo.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from device import device
from collections import namedtuple
from ppo_base import PPOBase
import sil
import numpy as np
import random


class Cells:
    def __init__(self, cell_num, cell_size, batch_size, data=None):
        self.cell_num = cell_num
        self.cell_size = cell_size
        self.batch_size = batch_size
        if data is None:
            self.data = [torch.zeros(batch_size, cell_size, device=device) for _ in range(cell_num)]
        else:
            self.data = data

    def clone(self):
        data = [cell.detach().clone() for cell in self.data]
        return Cells(self.cell_num, self.cell_size, self.batch_size, data)

    def get_masked(self, mask):
        mask = mask.view(-1, 1).expand_as(self.data[0]).to(device)
        return [cell * mask for cell in self.data]

    def reset(self):
        self.data = [cell.detach() for cell in self.data]

    def sub_range(self, r1, r2):
        data = [cell[r1:r2] for cell in self.data]
        return Cells(self.cell_num, self.cell_size, r2-r1, data)


class BaseModel(nn.Module):
    def __init__(self, in_channels, button_num, variable_num, frame_num, batch_size):
        super(BaseModel, self).__init__()
        self.screen_feature_num = 256
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)

        #self.screen_features1 = nn.Linear(512 * 2 * 4, self.screen_feature_num)
        self.screen_features1 = nn.LSTMCell(512 * 2 * 4 + variable_num + button_num, self.screen_feature_num)

        self.batch_norm = nn.BatchNorm1d(self.screen_feature_num)

        #variable_num = 0

        layer1_size = 128
        self.action1 = nn.Linear(self.screen_feature_num, layer1_size)
        self.action2 = nn.Linear(layer1_size, button_num)

        self.value1 = nn.Linear(self.screen_feature_num, layer1_size)
        self.value2 = nn.Linear(layer1_size, 1)

        self.screens = None
        self.frame_num = frame_num
        self.batch_size = batch_size
        self.button_num = button_num

    def forward(self, screen, variables, prev_action, cells, non_terminal, update_cells=True):
        # cnn
        screen_features = F.relu(self.conv1(screen))
        screen_features = F.relu(self.conv2(screen_features))
        screen_features = F.relu(self.conv3(screen_features))
        screen_features = F.relu(self.conv4(screen_features))
        screen_features = F.relu(self.conv5(screen_features))
        screen_features = F.relu(self.conv6(screen_features))
        screen_features = screen_features.view(screen_features.size(0), -1)
        screen_features = torch.cat([screen_features, variables, prev_action], 1)

        # rnn
        if screen_features.shape[0] <= self.batch_size:
            data = cells.get_masked(non_terminal)
            data = self.screen_features1(screen_features, data)
            if update_cells:
                cells.data = data
            return data[0]
        else:
            features = []
            for i in range(screen_features.shape[0]//self.batch_size):
                data = cells.get_masked(non_terminal[i])
                start = i * cells.batch_size
                cells.data = self.screen_features1(screen_features[start:start + cells.batch_size], data)
                features.append(cells.data[0])
            features = torch.cat(features, dim=0)
            return features

        #features = self.screen_features1(screen_features)
        #features = self.batch_norm(features)
        #features = F.relu(h1)

    def get_action(self, features):
        action = F.relu(self.action1(features))
        action = self.action2(action)
        return action

    def get_value(self, features):
        value = F.relu(self.value1(features))
        value = self.value2(value)
        return value

    def transform_input(self, screen, variables, prev_action):
        screen_batch = []
        if self.frame_num > 1:
            if self.screens is None:
                self.screens = [[]] * len(screen)
            for idx, screens in enumerate(self.screens):
                if len(screens) >= self.frame_num:
                    screens.pop(0)
                screens.append(screen[idx])
                if len(screens) == 1:
                    for i in range(self.frame_num - 1):
                        screens.append(screen[idx])
                screen_batch.append(torch.cat(screens, 0))
            screen = torch.stack(screen_batch)

        prev_action = torch.zeros(prev_action.shape[0], self.button_num, device=device).scatter(-1, prev_action.long(), 1)
        return screen.to(device), variables.to(device), prev_action

    def sil_transform_input(self, screen, variables, prev_action):
        screen_batch = []
        if self.frame_num > 1:
            if self.screens is None:
                self.screens = [[]] * len(screen)
            for idx, screens in enumerate(self.screens):
                if len(screens) >= self.frame_num:
                    screens.pop(0)
                screens.append(screen[idx])
                if len(screens) == 1:
                    for i in range(self.frame_num - 1):
                        screens.append(screen[idx])
                screen_batch.append(torch.cat(screens, 0))
            screen = torch.stack(screen_batch)

        prev_action = torch.zeros(prev_action.shape[0], self.button_num, device=device).scatter(-1, prev_action.long(), 1)
        return screen, variables, prev_action

    def set_non_terminal(self, non_terminal):
        if self.screens is not None:
            indexes = torch.nonzero(non_terminal == 0).squeeze()
            for idx in range(len(indexes)):
                self.screens[indexes[idx]] = []


StepInfo = namedtuple('StepInfo', ['screen', 'variables', 'prev_action', 'log_action', 'value', 'action'])


class PPO(PPOBase):
    def __init__(self, args):
        self.model = BaseModel(
            args.screen_size[0]*args.frame_num, args.button_num, args.variable_num, args.frame_num, args.batch_size
        ).to(device)
        if args.load is not None:
            # load weights
            state_dict = torch.load(args.load)
            self.model.load_state_dict(state_dict)

        self.discount = args.episode_discount
        self.steps = []
        self.rewards = []
        self.non_terminals = []
        self.non_terminal = torch.ones(args.batch_size, 1)
        self.buffer = sil.ReplayBuffer(100000)
        self.running_episodes = [[] for _ in range(args.batch_size)]
        self.n_env = args.batch_size
        self.total_steps = []
        self.total_rewards = []
        self.max_steps = 100000
        self.button_num = args.button_num
        self.fillnum = 0

        self.cells = Cells(2, self.model.screen_feature_num, args.batch_size)
        self.init_cells = self.cells.clone()

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate,  weight_decay=1e-6, amsgrad=True)
        '''
        if args.load is not None and os.path.isfile(args.load + '_optimizer.pth'):
            optimizer_dict = torch.load(args.load+'_optimizer.pth')
            optimizer.load_state_dict(optimizer_dict)
        '''
        self.optimizer.zero_grad()
        self.args = args

    def reset(self):
        self.steps = []
        self.rewards = []
        self.non_terminal = self.non_terminals[-1]
        self.non_terminals = []
        self.cells.reset()
        self.init_cells = self.cells.clone()

    def forward(self, screen, variables, prev_action, non_terminals, action_only=False, save_step_info=False, action=None, action_dist=False):
        features = self.model.forward(screen, variables, prev_action, self.cells, non_terminals)
        # print(screen.size(), variables.size(), prev_action.size(), non_terminals.size(), features.size())
        action_prob = self.model.get_action(features)

        if action_only:
            _, action = action_prob.max(1, keepdim=True)
            return action, None, None

        action_prob = F.softmax(action_prob, dim=1)

        if action is None:
            action = torch.multinomial(action_prob, 1)
            # greedy actions
            '''
            if random.random() < 0.01:
                action = torch.LongTensor(action_prob.size(0), 1).random_(0, action_prob.size(1)).to(device)
            else:
               _, action = action_prob.max(1, keepdim=True)
            '''

        # value prediction - critic
        value = self.model.get_value(features)
        # policy log
        #action_log_prob = action_prob.gather(-1, action).log()
        logits = action_prob.log()
        action_log_prob = logits.gather(-1, action)

        entropy = -(logits * action_prob).sum(-1)

        if save_step_info:
            # save step info for backward pass
            self.steps.append(StepInfo(screen, variables, prev_action, action_log_prob, value, action))

        return action, action_log_prob, value, entropy

    def get_action(self, state, prev_action, action_dist=False):
        with torch.set_grad_enabled(False):
            action, _, _ = self.forward(
                *self.model.transform_input(state.screen, state.variables, prev_action), self.non_terminal, action_only=True, action_dist=action_dist
            )
        return action

    def get_save_action(self, state, prev_action):
        with torch.set_grad_enabled(False):
            action, _, _, _ = self.forward(
                *self.model.transform_input(state.screen, state.variables, prev_action), self.non_terminal, save_step_info=True
            )
        return action

    def get_sil_action_value(self, screen, variables, prev_action, non_terminal):
        action, _,  value, _ = self.forward(screen, variables, prev_action, non_terminal)
        return action, value


    def set_last_state(self, state, prev_action):
        screen, variables, prev_action = self.model.transform_input(state.screen, state.variables, prev_action)
        with torch.set_grad_enabled(False):
            features = self.model.forward(
                *self.model.transform_input(state.screen, state.variables, prev_action),
                self.cells, self.non_terminal, update_cells=False
            )
            value = self.model.get_value(features)
        self.steps.append(StepInfo(None, None, None, None, value, None))

    def set_reward(self, reward):
        self.rewards.append(reward * 0.01)  # no clone() b/c of * 0.01

    def set_non_terminal(self, non_terminal):
        non_terminal = non_terminal.clone()
        self.model.set_non_terminal(non_terminal)
        self.non_terminals.append(non_terminal)
        self.non_terminal = non_terminal

    def backward(self, epi):
        rewards = self.rewards
        episode_steps = self.steps
        non_terminals = self.non_terminals
        saved_cells = self.cells.clone()

        #
        # calculate step returns in reverse order
        returns = torch.Tensor(len(rewards), *episode_steps[-1].value.shape)
        # last step contains only value, take it and delete the step
        step_return = episode_steps[-1].value.detach().cpu()
        del episode_steps[-1]
        for i in range(len(rewards) - 1, -1, -1):
            step_return.mul_(non_terminals[i]).mul_(self.discount).add_(rewards[i])
            returns[i] = step_return
        returns = returns.to(device)

        #
        # calculate advantage
        steps = len(episode_steps)
        advantage = torch.Tensor(*returns.shape)
        for i in range(steps):
            advantage[i] = returns[i] - episode_steps[i].value.detach()
        advantage = advantage.view(-1, 1).to(device)
        #print(advantage.mean().item(), advantage.min().item(), advantage.max().item())

        self.model.train()

        screens = torch.cat([step.screen for step in self.steps], dim=0)
        variables = torch.cat([step.variables for step in self.steps], dim=0)
        prev_actions = torch.cat([step.prev_action for step in self.steps], dim=0)
        non_terminals = torch.cat(self.non_terminals, dim=0)
        actions = torch.cat([step.action for step in self.steps], dim=0)
        old_log_actions = torch.cat([step.log_action for step in self.steps], dim=0)
        for batch in range(4):
            self.cells = self.init_cells.clone()

            _, log_actions, values, entropy = self.forward(screens, variables, prev_actions, non_terminals, action=actions)

            ratio = (log_actions - old_log_actions).exp()
            policy_loss = - torch.min(ratio * advantage, torch.clamp(ratio, 1 - 0.1, 1 + 0.1) * advantage).mean()
            value_loss = F.smooth_l1_loss(values, returns.view(-1, 1))
            entropy_loss = -entropy.mean() * 0.01

            #weights_l2 = 0
            #for param in self.parameters():
            #    weights_l2 += param.norm(2)

            loss = policy_loss + value_loss + entropy_loss #+ 0.0001*weights_l2
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            grads = []
            weights = []
            for p in self.model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
                    weights.append(p.view(-1))
            grads = torch.cat(grads, 0)
            weights = torch.cat(weights, 0)
            grads_norm = grads.norm()
            weights_norm = weights.norm()

            # check for NaN
            assert grads_norm == grads_norm

            self.optimizer.step()
            self.optimizer.zero_grad()

        # reset state
        if epi < 500:
            self.cells = saved_cells
            self.reset()

        return grads_norm, weights_norm, loss

    def silbackward(self):

        # rewards = self.rewards
        # episode_steps = self.steps
        # non_terminals = self.non_terminals
        saved_cells = self.cells.clone()
        batch_size = 20
        screens, action_ph, returns, prev_actions, variables, non_terminal = self.buffer.sample(batch_size)
        buffersize = self.buffer.get_buffersize()
        alp = 0.5
        self.model.train()

        for batch in range(4):
            self.cells = self.init_cells.clone()
            # compute an action
            # print(screens.size())
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            screens = screens.to(device)
            variables = variables.to(device)
            prev_actions = prev_actions.to(device)
            non_terminal = non_terminal.to(device)

            action_pred, value = self.get_sil_action_value(screens, variables, prev_actions, non_terminal)

            value = value.type(torch.FloatTensor)
            value_loss = F.smooth_l1_loss(returns.view(-1, 1), value)
            # print(action_pred.type(),action_pred.size())
            # print(action_ph.type(),action_ph.size())

            action_pred = torch.zeros(action_pred.shape[0], self.button_num, device=device).scatter(-1,
                                                                                                    action_pred.long(),
                                                                                                    1)
            # action_ph = action_ph.type(torch.LongTensor)

            # print(action_pred.type(),action_pred.size())
            # print(action_ph.squeeze().type(),action_ph.squeeze().size())
            actions_loss = nn.CrossEntropyLoss()
            output = actions_loss(action_pred, action_ph.squeeze())
            # actions_loss = torch.nn.CrossEntropyLoss(action_pred, action_ph)

            loss = value_loss + alp*output  # loss for sil

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            grads = []
            weights = []
            for p in self.model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
                    weights.append(p.view(-1))
            grads = torch.cat(grads, 0)
            weights = torch.cat(weights, 0)
            grads_norm = grads.norm()
            weights_norm = weights.norm()

            # check for NaN
            assert grads_norm == grads_norm

            self.optimizer.step()
            self.optimizer.zero_grad()

        # reset state
        self.cells = saved_cells
        self.reset()

        return grads_norm, weights_norm, loss, buffersize

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self):
        torch.save(self.model.state_dict(), self.args.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.args.checkpoint_file + '_optimizer.pth')


    def sil_step(self, obs, actions, rewards, prev_actions, variables, dones):
        obs, variables, prev_actions = self.model.sil_transform_input(obs, variables, prev_actions)
        for n in range(self.n_env):
            self.running_episodes[n].append([obs[n,:], actions[n,:], rewards[n,:], prev_actions[n,:], variables[n,:], dones[n,:]])
        for n, done in enumerate(dones):
            if done:
                self.update_buffer(self.running_episodes[n])
                self.running_episodes[n] = []

    def update_buffer(self, trajectory):
        positive_reward = False
        if self.fillnum < 1:
            for (ob, a, r, _, _, _) in trajectory:
                if r > 10:
                    positive_reward = True
                    self.fillnum += 1
                    break
        for (ob, a, r, _, _ ,_ ) in trajectory:
            if r > 10:
                positive_reward = True
                break
        if positive_reward:
            self.add_episode(trajectory)
            # self.total_steps.append(len(trajectory))
            # self.total_rewards.append(np.sum([x[2] for x in trajectory]))
            # while np.sum(self.total_steps) > self.max_steps and len(self.total_steps) > 1:
            #     self.total_steps.pop(0)
            #     self.total_rewards.pop(0)

    def add_episode(self, trajectory):
        screens = []
        actions = []
        rewards = []
        prev_actions = []
        varibales = []
        dones = []
        for (obs, action, reward, prev_action, variable, done) in trajectory:
            screens.append(obs)
            actions.append(action)
            rewards.append(reward)
            prev_actions.append(prev_action)
            varibales.append(variable)
            dones.append(done)

        returns = self.discount_with_dones(rewards, dones, self.discount)
        for (ob, action, R, preact, var, d) in list(zip(screens, actions, returns, prev_actions, varibales, dones)):
            self.buffer.add(ob, action, R, preact, var, d)

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)  # fixed off by one bug
            discounted.append(r)
        return discounted[::-1]
