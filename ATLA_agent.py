import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.99
EPS = 0.2  # PPO2 epsilon

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.fc1_actor = nn.Linear(1, FEATURE_NUM)
        self.fc2_actor = nn.Linear(1, FEATURE_NUM)
        self.conv1_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv2_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv3_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv4_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv5_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv6_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv7_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.fc4_actor = nn.Linear(FEATURE_NUM * self.s_dim[0], FEATURE_NUM)
        self.dist_fc = nn.Linear(FEATURE_NUM, FEATURE_NUM)
        self.alpha_head = nn.Linear(FEATURE_NUM, 1)
        self.beta_head = nn.Linear(FEATURE_NUM, 1)

    def forward(self, inputs):
        split_0 = F.relu(self.fc1_actor(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc2_actor(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv1_actor(inputs[:, 2:3, :]).view(-1, FEATURE_NUM))
        split_3 = F.relu(self.conv2_actor(inputs[:, 3:4, :]).view(-1, FEATURE_NUM))
        split_4 = F.relu(self.conv3_actor(inputs[:, 4:5, :]).view(-1, FEATURE_NUM))
        split_5 = F.relu(self.conv4_actor(inputs[:, 5:6, :]).view(-1, FEATURE_NUM))
        split_6 = F.relu(self.conv5_actor(inputs[:, 6:7, :]).view(-1, FEATURE_NUM))
        split_7 = F.relu(self.conv6_actor(inputs[:, 7:8, :]).view(-1, FEATURE_NUM))
        split_8 = F.relu(self.conv7_actor(inputs[:, 8:9, :]).view(-1, FEATURE_NUM))

        merge_net = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5, split_6, split_7, split_8], 1)

        h = F.relu(self.fc4_actor(merge_net))
        h = F.relu(self.dist_fc(h))
        alpha = F.softplus(self.alpha_head(h)) + 1e-3
        beta = F.softplus(self.beta_head(h)) + 1e-3
        return alpha, beta


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.fc1_actor = nn.Linear(1, FEATURE_NUM)
        self.fc2_actor = nn.Linear(1, FEATURE_NUM)
        self.conv1_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv2_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv3_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv4_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv5_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv6_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv7_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.fc4_actor = nn.Linear(FEATURE_NUM * self.s_dim[0], FEATURE_NUM)
        self.val_head = nn.Linear(FEATURE_NUM, 1)

    def forward(self, inputs):
        split_0 = F.relu(self.fc1_actor(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc2_actor(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv1_actor(inputs[:, 2:3, :]).view(-1, FEATURE_NUM))
        split_3 = F.relu(self.conv2_actor(inputs[:, 3:4, :]).view(-1, FEATURE_NUM))
        split_4 = F.relu(self.conv3_actor(inputs[:, 4:5, :]).view(-1, FEATURE_NUM))
        split_5 = F.relu(self.conv4_actor(inputs[:, 5:6, :]).view(-1, FEATURE_NUM))
        split_6 = F.relu(self.conv5_actor(inputs[:, 6:7, :]).view(-1, FEATURE_NUM))
        split_7 = F.relu(self.conv6_actor(inputs[:, 7:8, :]).view(-1, FEATURE_NUM))
        split_8 = F.relu(self.conv7_actor(inputs[:, 8:9, :]).view(-1, FEATURE_NUM))

        merge_net = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5, split_6, split_7, split_8], 1)

        value_net = F.relu(self.fc4_actor(merge_net))
        value = self.val_head(value_net)
        return value
    
class Network():
    def __init__(self, state_dim, action_dim, learning_rate):

        self.s_dim = state_dim
        self.action_dim = action_dim
        self._entropy_weight = np.log(action_dim)
        self.H_target = 0.1
        self.PPO_TRAINING_EPO = 5

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.lr_rate = learning_rate
        self.optimizer = optim.Adam(list(self.actor.parameters()) + \
                                    list(self.critic.parameters()), lr=learning_rate)

    def get_network_params(self):
        return [self.actor.state_dict(), self.critic.state_dict()]
    
    def set_network_params(self, input_network_params):
        actor_net_params, critic_net_params = input_network_params
        self.actor.load_state_dict(actor_net_params)
        self.critic.load_state_dict(critic_net_params)

    def r_beta(self, alpha_new, beta_new, alpha_old, beta_old, x):
        lgamma_new = torch.lgamma(alpha_new) + torch.lgamma(beta_new) - torch.lgamma(alpha_new + beta_new)
        lgamma_old = torch.lgamma(alpha_old) + torch.lgamma(beta_old) - torch.lgamma(alpha_old + beta_old)
        log_prob_new = (alpha_new - 1.0) * torch.log(x + 1e-8) + (beta_new - 1.0) * torch.log(1.0 - x + 1e-8) - lgamma_new
        log_prob_old = (alpha_old - 1.0) * torch.log(x + 1e-8) + (beta_old - 1.0) * torch.log(1.0 - x + 1e-8) - lgamma_old
        return torch.exp(log_prob_new - log_prob_old)

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch):
        s_batch = torch.from_numpy(s_batch).to(torch.float32).to(self.device)
        x_batch = torch.from_numpy(a_batch).to(torch.float32).to(self.device)  # shape (B,1)
        p_batch = torch.from_numpy(p_batch).to(torch.float32).to(self.device)  # shape (B,2) [alpha, beta]
        v_batch = torch.from_numpy(v_batch).to(torch.float32).to(self.device)

        total_loss = 0.0
        for _ in range(self.PPO_TRAINING_EPO):
            alpha_new, beta_new = self.actor.forward(s_batch)
            val = self.critic.forward(s_batch)

            # loss
            adv = v_batch - val.detach()
            alpha_old = p_batch[:, 0:1]
            beta_old = p_batch[:, 1:2]
            ratio = self.r_beta(alpha_new, beta_new, alpha_old, beta_old, x_batch)
            ppo2loss = torch.min(ratio * adv, torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv)
            # Dual-PPO
            dual_loss = torch.where(adv < 0, torch.max(ppo2loss, 3. * adv), ppo2loss)
            # Beta distribution entropy approximation
            H = (torch.lgamma(alpha_new) + torch.lgamma(beta_new) - torch.lgamma(alpha_new + beta_new)) \
                - (alpha_new - 1) * torch.digamma(alpha_new) \
                - (beta_new - 1) * torch.digamma(beta_new) \
                + (alpha_new + beta_new - 2) * torch.digamma(alpha_new + beta_new)

            loss = -dual_loss.mean() + 10. * F.mse_loss(val, v_batch) - self._entropy_weight * H.mean()
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update entropy weight
        _g = H.mean().item() - self.H_target
        self._entropy_weight -= self.lr_rate * _g * 0.1 * self.PPO_TRAINING_EPO
        self._entropy_weight = max(self._entropy_weight, 1e-2)
        
        # Return average loss and current entropy
        avg_loss = total_loss / self.PPO_TRAINING_EPO
        return avg_loss, H.mean().item()

    def predict(self, input):
        with torch.no_grad():
            input = torch.from_numpy(input).to(torch.float32).to(self.device)
            alpha, beta = self.actor.forward(input)
            value = self.critic.forward(input)
            params = torch.cat([alpha, beta], dim=1)
            return params.cpu().numpy(), value.cpu().numpy()

    def predict_adv(self, input):
        alpha, beta = self.actor.forward(input)
        value = self.critic.forward(input)
        return (alpha, beta), value


    def load_model(self, nn_model):
        actor_model_params, critic_model_params = torch.load(nn_model, map_location=torch.device('cpu'))
        self.actor.load_state_dict(actor_model_params)
        self.critic.load_state_dict(critic_model_params)

    def save_model(self, nn_model):
        model_params = [self.actor.state_dict(), self.critic.state_dict()]
        torch.save(model_params, nn_model)

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        R_batch = np.zeros_like(r_batch)

        if terminal:
            R_batch[-1] = r_batch[-1]
        else:
            if isinstance(s_batch, list):
                s_batch_np = np.stack(s_batch, axis=0)
            else:
                s_batch_np = np.array(s_batch)
            s_tensor = torch.from_numpy(s_batch_np).to(torch.float32).to(self.device)
            val = self.critic.forward(s_tensor)
            R_batch[-1] = val[-1].item()

        for t in reversed(range(len(r_batch) - 1)):
            R_batch[t] = r_batch[t] + GAMMA * R_batch[t + 1]

        return list(R_batch)
           
if __name__ == "__main__":
    net = Network(state_dim=[9, 8], action_dim=8, learning_rate=0.001)
    input = torch.rand(1,9,8).numpy()

    pi, value = net.predict(input)
    print(pi)
    print(value)
