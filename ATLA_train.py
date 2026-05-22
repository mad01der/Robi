import numpy as np
import os
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def flush(self):
            pass
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from env import ABREnv
import sys
import subprocess

import ATLA_adv 
import ATLA_agent 

import torch
import re

S_DIM = [9, 8]
A_DIM = 8
ACTOR_LR_RATE = 1e-4
TRAIN_SEQ_LEN = 500
TRAIN_EPOCH = 500000
MODEL_SAVE_INTERVAL = 30
RANDOM_SEED = 42
SUMMARY_DIR = './atla_test'
TEST_LOG_FOLDER = './atla_test_results_test/'
LOG_FILE = SUMMARY_DIR + '/atla_log'

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

ADV_NN_MODEL = None
AGENT_NN_MODEL = None

def testing(epoch, nn_model, log_file):
    # 为每个模型创建单独的测试结果文件夹并运行测试
    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    model_base = os.path.splitext(os.path.basename(nn_model))[0]
    model_out_dir = os.path.join(TEST_LOG_FOLDER, model_base)
    os.makedirs(model_out_dir, exist_ok=True)
    try:
        test_dir = os.path.abspath('./train/huawei500kbps/test/')
        subprocess.run([sys.executable, 'atla_test.py', nn_model, test_dir, model_out_dir], check=True)
    except Exception:
        pass

    # append test performance to the log
    rewards, entropies = [], []
    test_log_files = os.listdir(model_out_dir)
    if len(test_log_files) == 0:
        log_file.write(str(epoch) + '\t' +
                       str(0.0) + '\t' +
                       str(0.0) + '\t' +
                       str(0.0) + '\t' +
                       str(0.0) + '\t' +
                       str(0.0) + '\t' +
                       str(0.0) + '\n')
        log_file.flush()
        return 0.0, 0.0
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(os.path.join(model_out_dir, test_log_file), 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        if len(reward) > 1 and len(entropy) > 1:
            rewards.append(np.mean(reward[1:]))
            entropies.append(np.mean(entropy[1:]))

    if len(rewards) == 0:
        log_file.write(str(epoch) + '\t' +
                       str(0.0) + '\t' +
                       str(0.0) + '\t' +
                       str(0.0) + '\t' +
                       str(0.0) + '\t' +
                       str(0.0) + '\t' +
                       str(0.0) + '\n')
        log_file.flush()
        return 0.0, 0.0
    rewards = np.array(rewards)
    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()

    return rewards_mean, np.mean(entropies)
        
def _find_latest_model(dir_path, prefix):
    best_epoch = -1
    best_path = None
    try:
        files = os.listdir(dir_path)
    except Exception:
        files = []
    for name in files:
        if not name.startswith(prefix) or not name.endswith('.pth'):
            continue
        try:
            m = re.search(rf'{re.escape(prefix)}(\d+)\.pth$', name)
            if not m:
                continue
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best_path = os.path.join(dir_path, name)
        except Exception:
            continue
    return best_path, best_epoch

def main():
    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    
    env = ABREnv(0)
    adversary = ATLA_adv.Network(state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
    agent = ATLA_agent.Network(state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
    # 加速训练：减少每轮PPO迭代次数
    agent.PPO_TRAINING_EPO = 3
    adversary.PPO_TRAINING_EPO = 3
    
    adv_nn_model = ADV_NN_MODEL
    if adv_nn_model is not None:
        adversary.load_model(adv_nn_model)
        print('adv_nn_model 模型已恢复.')

    agent_nn_model = AGENT_NN_MODEL
    if agent_nn_model is not None:
        agent.load_model(agent_nn_model)
        print('agent_nn_model 模型已恢复.')

    # 初始化TensorBoard写入器
    writer = SummaryWriter(SUMMARY_DIR)
    # 打开测试日志文件
    with open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        torch.autograd.set_detect_anomaly(False)
        latest_agent_path, latest_agent_epoch = _find_latest_model(SUMMARY_DIR, 'agent_nn_model_ep_')
        latest_adv_path, latest_adv_epoch = _find_latest_model(SUMMARY_DIR, 'adv_nn_model_ep_')
        start_epoch = 0
        latest_agent_fixed = os.path.join(SUMMARY_DIR, 'agent_nn_model_latest.pth')
        latest_adv_fixed = os.path.join(SUMMARY_DIR, 'adv_nn_model_latest.pth')
        if os.path.exists(latest_agent_fixed):
            try:
                agent.load_model(latest_agent_fixed)
                start_epoch = max(start_epoch, 1)
                print(f'恢复 agent_nn_model: {latest_agent_fixed}')
            except Exception:
                pass
        if latest_agent_path:
            try:
                agent.load_model(latest_agent_path)
                start_epoch = max(start_epoch, latest_agent_epoch + 1)
                print(f'恢复 agent_nn_model: {latest_agent_path}')
            except Exception:
                pass
        if os.path.exists(latest_adv_fixed):
            try:
                adversary.load_model(latest_adv_fixed)
                start_epoch = max(start_epoch, 1)
                print(f'恢复 adv_nn_model: {latest_adv_fixed}')
            except Exception:
                pass
        if latest_adv_path:
            try:
                adversary.load_model(latest_adv_path)
                start_epoch = max(start_epoch, latest_adv_epoch + 1)
                print(f'恢复 adv_nn_model: {latest_adv_path}')
            except Exception:
                pass
        try:
            for epoch in range(start_epoch, TRAIN_EPOCH):
                obs = env.reset()
                s_batch, a_batch, p_batch, r_batch = [], [], [], []

                for param in agent.actor.parameters():
                    param.requires_grad = True

                for param in agent.critic.parameters():
                    param.requires_grad = True

                for step in range(TRAIN_SEQ_LEN):
                    s_batch.append(obs)
                    params, _ = agent.predict(np.reshape(obs, (1, S_DIM[0], S_DIM[1])))
                    alpha, beta = params[0, 0], params[0, 1]
                    x = np.random.beta(alpha, beta)
                    obs, rew, done, info = env.step(x)
                    a_batch.append([x])
                    r_batch.append(rew)
                    p_batch.append([alpha, beta])
                    if done:
                        break
                
                v_batch = agent.compute_v(s_batch, a_batch, r_batch, done)
                s_batch = np.stack(s_batch, axis=0)
                perturbed_obs = adversary.predict(s_batch)
                a_batch = np.array(a_batch, dtype=np.float32)
                p_batch = np.array(p_batch, dtype=np.float32)
                v_batch = np.vstack(v_batch)

                agent_loss, agent_entropy = agent.train(perturbed_obs, a_batch, p_batch, v_batch, epoch)
                avg_reward_phase1 = np.mean(r_batch)
                print(f"Epoch {epoch} Phase1 - Avg Reward: {avg_reward_phase1:.4f}, Agent Loss: {agent_loss:.4f}, Entropy: {agent_entropy:.4f}")
                if epoch > 0 and epoch % MODEL_SAVE_INTERVAL == 0:
                    model_path = SUMMARY_DIR + '/agent_nn_model_ep_' + str(epoch) + '.pth'
                    agent.save_model(model_path)
                    print("开始测试模型...")
                    avg_reward, avg_entropy = testing(epoch, model_path, test_log_file)
                    writer.add_scalar('Entropy Weight', agent._entropy_weight, epoch)
                    writer.add_scalar('Reward', avg_reward, epoch)
                    writer.add_scalar('Entropy', avg_entropy, epoch)
                    writer.flush()
                    print(f"Epoch {epoch}, Test Avg Reward: {avg_reward:.4f}, Test Avg Entropy: {avg_entropy:.4f}")



                obs = env.reset()
                s_batch, a_batch, p_batch, r_batch = [], [], [], []


                for step in range(TRAIN_SEQ_LEN):
                    s_batch.append(obs)
                    params, _ = agent.predict(np.reshape(obs, (1, S_DIM[0], S_DIM[1])))
                    alpha, beta = params[0, 0], params[0, 1]
                    x = np.random.beta(alpha, beta)
                    obs, rew, done, info = env.step(x)

                
                a_batch.append([x])
                r_batch.append(rew)
                p_batch.append([alpha, beta])
                if done:
                    break
            
                v_batch = agent.compute_v(s_batch, a_batch, r_batch, done)
                s_batch = np.stack(s_batch, axis=0)
                a_batch = np.array(a_batch, dtype=np.float32)
                p_batch = np.array(p_batch, dtype=np.float32)
                v_batch = np.vstack(v_batch)
                for param in agent.actor.parameters():
                    param.requires_grad = False

                for param in agent.critic.parameters():
                    param.requires_grad = False

                adv_loss = adversary.train(s_batch, a_batch, p_batch, v_batch, agent, epoch)
                avg_reward_phase2 = np.mean(r_batch)
                print(f"Epoch {epoch} Phase2 - Avg Reward: {avg_reward_phase2:.4f}, Adv Loss: {adv_loss:.4f}")
                if epoch % MODEL_SAVE_INTERVAL == 0:
                    model_path = SUMMARY_DIR + '/adv_nn_model_ep_' + str(epoch) + '.pth'
                    adversary.save_model(model_path)
                    print(f"   adv_loss: {adv_loss}")
                try:
                    agent.save_model(SUMMARY_DIR + '/agent_nn_model_latest.pth')
                except Exception:
                    pass
                try:
                    adversary.save_model(SUMMARY_DIR + '/adv_nn_model_latest.pth')
                except Exception:
                    pass
                writer.flush()
        except KeyboardInterrupt:
            try:
                agent.save_model(SUMMARY_DIR + '/agent_nn_model_ep_' + str(epoch) + '.pth')
            except Exception:
                pass
            try:
                adversary.save_model(SUMMARY_DIR + '/adv_nn_model_ep_' + str(epoch) + '.pth')
            except Exception:
                pass
            try:
                agent.save_model(SUMMARY_DIR + '/agent_nn_model_latest.pth')
            except Exception:
                pass
            try:
                adversary.save_model(SUMMARY_DIR + '/adv_nn_model_latest.pth')
            except Exception:
                pass
            print(f'训练已中断，已保存模型至 epoch {epoch}')


if __name__ == '__main__':
    main()
