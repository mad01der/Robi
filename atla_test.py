import os
import sys
import numpy as np
from env import ABREnv
import ATLA_agent as network
import load_trace


S_INFO = 9
S_LEN = 8
A_DIM = 8
ACTOR_LR_RATE = 0.0001
RANDOM_SEED = 42
NN_MODEL = sys.argv[1]
TEST_DIR = sys.argv[2] if len(sys.argv) > 2 else None
OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else './atla_test_results'
    
def main():

    np.random.seed(RANDOM_SEED)

    env = ABREnv(0, cooked_trace_folder=TEST_DIR)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)


    actor = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)

    # restore neural net parameters
    if NN_MODEL is not None:  # NN_MODEL is the path to file
        actor.load_model(NN_MODEL)
        print("Testing model restored.")

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_DIR)
    for idx, fname in enumerate(all_file_names):
        env.net_env.reset_to_trace(idx)
        time_stamp = 0.0
        log_path = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(fname))[0] + '.txt')
        log_file = open(log_path, 'w')
        obs = env.reset()
        prev_mbps = env._last_send_bitrate / 1000.0
        while True:
            params, _ = actor.predict(np.reshape(obs, (1, S_INFO, S_LEN)))
            alpha, beta = params[0, 0], params[0, 1]
            x = np.random.beta(alpha, beta)
            obs, reward, end_of_video, info = env.step(x)
            time_stamp += info['delay']
            smooth_mbps = abs(info['bitrate'] - prev_mbps)
            prev_mbps = info['bitrate']
            log_file.write(str(time_stamp) + '\t' +
                           str(info['bitrate']) + '\t' +
                           str(info['delay']) + '\t' +
                           str(info['packet_loss']) + '\t' +
                           str(smooth_mbps) + '\t' +
                           str(0.0) + '\t' +
                           str(reward) + '\n')
            log_file.flush()
            if end_of_video:
                log_file.write('\n')
                log_file.close()
                break


if __name__ == '__main__':
    main()
