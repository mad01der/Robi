import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from env import ABREnv

FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.99
EPS = 0.2  # PPO2 epsilon

S_INFO = 9
S_LEN = 8  # take how many frames in the past

# class AdversaryNetwork(nn.Module):
#     def __init__(self, state_dim, hidden_dim=128, perturbation_bound=0.1):
#         super(AdversaryNetwork, self).__init__()
#         self.perturbation_bound = perturbation_bound
        
#         self.fc1 = nn.Linear(state_dim * S_LEN, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.perturbation = nn.Linear(hidden_dim, state_dim * S_LEN)
        
#         # 初始化
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
#                 nn.init.constant_(m.bias, 0)
        
#         nn.init.orthogonal_(self.perturbation.weight, gain=0.01)
#         nn.init.constant_(self.perturbation.bias, 0)
        
#     def forward(self, x):
#         # print(x)
#         batch_size = x.size(0)
#         x_flat = x.view(batch_size, -1)
        
#         x1 = torch.relu(self.fc1(x_flat))
#         x2 = torch.relu(self.fc2(x1))
#         x3 = torch.tanh(self.perturbation(x2))
#         # 生成扰动
#         perturbation = (x3+1) * self.perturbation_bound
        
#         # 应用扰动
#         perturbed_x = x_flat + perturbation
#         output = perturbed_x.view(batch_size, S_INFO, S_LEN)
#         # print(perturbed_x)
#         # 重塑回原始状态形状
#         return output



class AdversaryNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, perturbation_bound=0.01):
        super(AdversaryNetwork, self).__init__()
        # Actor network
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.perturbation_bound = perturbation_bound

        self.fc1_actor = nn.Linear(1, FEATURE_NUM)
        self.fc2_actor = nn.Linear(1, FEATURE_NUM)
        self.conv1_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv2_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv3_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv4_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv5_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv6_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv7_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.fc4_actor = nn.Linear(FEATURE_NUM , self.s_dim[1])

    def forward(self, inputs):
        # x = inputs
        split_0 = F.relu(self.fc1_actor(inputs[:, 0:1, -1:]))
        split_1 = F.relu(self.fc2_actor(inputs[:, 1:2, -1:]))
        split_2 = F.relu(self.conv1_actor(inputs[:, 2:3, :]))
        split_3 = F.relu(self.conv2_actor(inputs[:, 3:4, :]))
        split_4 = F.relu(self.conv3_actor(inputs[:, 4:5, :]))
        split_5 = F.relu(self.conv4_actor(inputs[:, 5:6, :]))
        split_6 = F.relu(self.conv5_actor(inputs[:, 6:7, :]))
        split_7 = F.relu(self.conv6_actor(inputs[:, 7:8, :]))
        split_8 = F.relu(self.conv7_actor(inputs[:, 8:9, :]))

        merge_net = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5, split_6, split_7, split_8], 1)

        noise = F.tanh(self.fc4_actor(merge_net))
        noise_nor = torch.add(noise, 1)
        noise_nor2 = torch.multiply(noise_nor, self.perturbation_bound)
        output = torch.add(inputs, noise_nor2)
        return output



  
class Network():
    def __init__(self, state_dim, action_dim, learning_rate):

        self.s_dim = state_dim
        self.H_target = 0.1
        self.PPO_TRAINING_EPO = 5
        self._entropy_weight = np.log(action_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adv = AdversaryNetwork(state_dim, action_dim=action_dim).to(self.device)
        self.lr_rate = learning_rate
        self.optimizer = optim.Adam(self.adv.parameters() , lr=learning_rate)

    def get_network_params(self):
        return self.adv.state_dict()
    
    def set_network_params(self, adv_network_params):
        self.adv.load_state_dict(adv_network_params)

    def r(self, pi_new, pi_old, acts):
        return torch.sum(pi_new * acts, dim=1, keepdim=True) / \
               torch.sum(pi_old * acts, dim=1, keepdim=True)

    def train(self, s_batch, a_batch, p_batch, v_batch, agent, epoch):
        s_batch = torch.from_numpy(s_batch).to(torch.float32).to(self.device)
        perturbed_s_batch  = self.adv(s_batch)
        x_batch = torch.from_numpy(a_batch).to(torch.float32).to(self.device)  # shape (B,1)
        p_batch = torch.from_numpy(p_batch).to(torch.float32).to(self.device)  # shape (B,2)
        v_batch = torch.from_numpy(v_batch).to(torch.float32).to(self.device)

        # for param in agent.actor.parameters():
        #     param.requires_grad = False

        # for param in agent.critic.parameters():
        #     param.requires_grad = False

        total_loss = 0.0
        for _ in range(self.PPO_TRAINING_EPO):

            perturbed_s_batch  = self.adv(s_batch)
            (alpha_new, beta_new), val = agent.predict_adv(perturbed_s_batch)
            # loss
            adv = v_batch - val
            alpha_old = p_batch[:, 0:1]
            beta_old = p_batch[:, 1:2]
            ratio = agent.r_beta(alpha_new, beta_new, alpha_old, beta_old, x_batch)

            ppo2loss = torch.min(ratio * adv, torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv)
            # Dual-PPO
            dual_loss = torch.where(adv < 0, torch.max(ppo2loss, 3. * adv), ppo2loss)
            H = (torch.lgamma(alpha_new) + torch.lgamma(beta_new) - torch.lgamma(alpha_new + beta_new)) \
                - (alpha_new - 1) * torch.digamma(alpha_new) \
                - (beta_new - 1) * torch.digamma(beta_new) \
                + (alpha_new + beta_new - 2) * torch.digamma(alpha_new + beta_new)

            loss = -(-dual_loss.mean() + 10. * F.mse_loss(val, v_batch) - self._entropy_weight * H.mean())
            total_loss += loss.item()

            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update entropy weight
        _H = (-(torch.log(p_batch) * p_batch).sum(dim=1)).mean().item()
        _g = _H - self.H_target
        self._entropy_weight -= self.lr_rate * _g * 0.1 * self.PPO_TRAINING_EPO
        self._entropy_weight = max(self._entropy_weight, 1e-2)
        
        # Return average adversarial loss
        avg_loss = total_loss / self.PPO_TRAINING_EPO
        return avg_loss

    def predict(self, input):
        with torch.no_grad():
            input = torch.from_numpy(input).to(torch.float32).to(self.device)
            perturbed_x = self.adv.forward(input)
            return perturbed_x.cpu().numpy()

    def load_model(self, nn_model):
        adv_model_params = torch.load(nn_model)
        self.adv.load_state_dict(adv_model_params)

    def save_model(self, nn_model):
        model_params = self.adv.state_dict()
        torch.save(model_params, nn_model)

    
if __name__ == "__main__":
    adv_ = AdversaryNetwork(state_dim=[9, 8], action_dim=8)

    adv = Network(state_dim=[9, 8], action_dim=8, learning_rate=0.001)

    env = ABREnv(random_seed=666)
    state = env.reset()
    state = np.expand_dims(np.array(state), axis=0)
    print(state)
    print(state.shape)
    print("\n")
    print(adv_(torch.Tensor(state)).shape)
    print(adv_(torch.Tensor(state)).detach())

    
