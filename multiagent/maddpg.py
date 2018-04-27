from multiagent import AgentTrainer
import common.tf_util as U


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, args, agent_index, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name='observation'+str(i)).get())

    def action(self, obs):
        pass

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        pass

    def preupdate(self):
        pass

    def update(self):
        pass
