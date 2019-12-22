from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# Definition of Message Encoder
class Env_blender(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_shape):
        super(Env_blender, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_shape)
        self.fc2 = nn.Linear(hidden_shape, output_shape)

    def forward(self, input_mac):
        hidden = F.elu(self.fc1(input_mac))
        output_mac = self.fc2(hidden)
        return output_mac


# Agent Network
class BasicMAC_8m:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self._parallel_batch_size = args.batch_size_run
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.match_weight = 0.0
        self.env_blender = Env_blender(64, 14, 196).cuda()
        self.delta1 = self.args.delta1
        self.delta2 = self.args.delta2
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):

        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_local_outputs, hidden_states = self.forward(ep_batch, t_ep, test_mode=test_mode)
        msg_list = [self.env_blender(hidden_states[:, i, :].view(self._parallel_batch_size, -1))
                    for i in range(self.n_agents)]
        msg_list = th.stack(msg_list, 1)
        msg_sum = th.sum(msg_list, 0)
        msg_per_agent = (msg_list - msg_sum) / (self.n_agents - 1)

        # generate the action-value function of each agent
        agent_outputs = agent_local_outputs + msg_per_agent

        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions

    def select_actions_comm_proto(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_local_outputs, hidden_states = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # dummy0 = self.env_blender(hidden_states[:, 0, :].view(self._parallel_batch_size, -1))
        # dummy1 = self.env_blender(hidden_states[:, 1, :].view(self._parallel_batch_size, -1))
        # dummy2 = self.env_blender(hidden_states[:, 2, :].view(self._parallel_batch_size, -1))
        # dummy3 = self.env_blender(hidden_states[:, 3, :].view(self._parallel_batch_size, -1))
        # dummy4 = self.env_blender(hidden_states[:, 4, :].view(self._parallel_batch_size, -1))
        # dummy5 = self.env_blender(hidden_states[:, 5, :].view(self._parallel_batch_size, -1))
        #
        # agent0 = (dummy1 + dummy2 + dummy3 + dummy4 + dummy5)/5.0
        # agent1 = (dummy0 + dummy2 + dummy3 + dummy4 + dummy5)/5.0
        # agent2 = (dummy0 + dummy1 + dummy3 + dummy4 + dummy5)/5.0
        # agent3 = (dummy0 + dummy1 + dummy2 + dummy4 + dummy5)/5.0
        # agent4 = (dummy0 + dummy1 + dummy2 + dummy3 + dummy5)/5.0
        # agent5 = (dummy0 + dummy1 + dummy2 + dummy3 + dummy4)/5.0
        msg_list = [self.env_blender(hidden_states[:, i, :].view(self._parallel_batch_size, -1))
                    for i in range(self.n_agents)]
        msg_list = th.stack(msg_list, 1)

        # elminate the communication based on the delta2

        num_eliminated_we = th.ones((self._parallel_batch_size, self.n_agents))
        for i in range(0, self._parallel_batch_size):

            q_std = th.std(msg_list[i], dim=1)
            high_var_idx = (self.delta2 > q_std).nonzero()
            if len(high_var_idx):
                num_eliminated_we[i, high_var_idx] = 0
                msg_list[i, high_var_idx] = 0
            # if(th.std(dummy0[i,:], dim=0)<self.delta2):
            #     dummy0[i:,] = th.zeros(1,14)
            #     num_eliminated_we[i,0] = 0
            # if(th.std(dummy1[i,:], dim=0)<self.delta2):
            #     dummy1[i:,] = th.zeros(1,14)
            #     num_eliminated_we[i,1] = 0
            # if(th.std(dummy2[i,:], dim=0)<self.delta2):
            #     dummy2[i:,] = th.zeros(1,14)
            #     num_eliminated_we[i,2] = 0
            # if(th.std(dummy3[i,:], dim=0)<self.delta2):
            #     dummy3[i:,] = th.zeros(1,14)
            #     num_eliminated_we[i,3] = 0
            # if(th.std(dummy4[i,:], dim=0)<self.delta2):
            #     dummy4[i:,] = th.zeros(1,14)
            #     num_eliminated_we[i,4] = 0
            # if(th.std(dummy5[i,:], dim=0)<self.delta2):
            #     dummy5[i:,] = th.zeros(1,14)
            #     num_eliminated_we[i,5] = 0
        # print(num_eliminated)
        num_eliminated = th.stack([num_eliminated_we.sum(1)] * self.n_agents).t() - num_eliminated_we
        msg_sum = th.sum(msg_list, 0)
        msg_per_agent = (msg_list - msg_sum) / (self.n_agents - 1)
        largest = th.topk(agent_local_outputs, 2, dim=2)
        # print('agent_local_outputs shape is: ' + str(agent_local_outputs.shape))
        # print('largest shape is: ' + str(largest[0]))
        criteria = th.abs(largest[0][:, :, 0] - largest[0][:, :, 1])
        binary_matrix = th.stack([(criteria > self.delta1).type(th.cuda.FloatTensor)] * 14,
                                 -1)  # duplicate along the third dimension
        # print(criteria.shape)
        # print(num_eliminated)
        # print((criteria < self.delta1).type(th.cuda.FloatTensor))
        # print('\n\n\n\n')
        comm_rate_local = ((criteria < self.delta1).type(th.cuda.FloatTensor)).sum() / (
                    self.n_agents * self._parallel_batch_size)
        comm_rate_among = (num_eliminated_we.cuda()).sum() / (self.n_agents * self._parallel_batch_size)
        comm_rate = (num_eliminated.cuda() * (criteria < self.delta1).type(th.cuda.FloatTensor)).sum() / \
                    (self.n_agents * (self.n_agents - 1) * self._parallel_batch_size)

        # generate the action-value function of each agent
        agent_outputs = agent_local_outputs * binary_matrix + (agent_local_outputs + msg_per_agent) * (
                    1 - binary_matrix)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)

        return chosen_actions, comm_rate.cpu().numpy(), comm_rate_local.cpu().numpy(), comm_rate_among.cpu().numpy()

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            print('does not enterring here!!!!!!!!!!!!!!')
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), self.hidden_states.view(ep_batch.batch_size,
                                                                                                self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def set_match_weight(self, weight):
        self.match_weight = weight

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.env_blender.load_state_dict(other_mac.env_blender.state_dict())
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.env_blender.state_dict(), "{}/env_blender.th".format(path))
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.env_blender.load_state_dict(
            th.load("{}/env_blender.th".format(path), map_location=lambda storage, loc: storage))
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
