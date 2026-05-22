# RTC environment for real-time video streaming
import os
import numpy as np
import core as abrenv
import load_trace

# bit_rate, buffer_size, next_frame_size, bandwidth_measurement(throughput and time), frame_til_video_end, packet_loss_rate
S_INFO = 9
S_LEN = 8
A_DIM = 8
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
MIN_BITRATE_KBPS = 80.0
MAX_BITRATE_KBPS = 500.0
DEFAULT_SEND_BITRATE_KBPS = 300.0
# Relative bitrate adjustment: x in [0,1] -> r in [-REL_DOWN_MAX, +REL_UP_MAX]
REL_DOWN_MAX = 0.5   # x=0: at most -50%
REL_UP_MAX = 0.2     # x=1: at most +20%
BUFFER_NORM_FACTOR = 0.1  # RTC uses smaller buffer normalization (100ms)
FRAME_TIL_VIDEO_END_CAP = 599.0  # Total frames in RTC scenario
M_IN_K = 1000.0
# RTC reward function coefficients
PACKET_LOSS_PENALTY = 10.0  # Penalty for packet loss
DELAY_PENALTY = 0.01  # Penalty for delay (ms)
SMOOTH_PENALTY = 0.001  # Penalty for bitrate smoothness
RANDOM_SEED = 42
RAND_RANGE = 1000
EPS = 1e-6


class ABREnv():
    def __init__(self, random_seed=RANDOM_SEED, cooked_trace_folder=None):
        np.random.seed(random_seed)
        if cooked_trace_folder is None:
            all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()
        else:
            all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(cooked_trace_folder)
        self.net_env = abrenv.Environment(all_cooked_time=all_cooked_time,
                                          all_cooked_bw=all_cooked_bw,
                                          random_seed=random_seed,
                                          min_bitrate_kbps=MIN_BITRATE_KBPS,
                                          max_bitrate_kbps=MAX_BITRATE_KBPS)

        self.last_bit_rate = 0
        self.buffer_size = 0.
        self.state = np.zeros((S_INFO, S_LEN))
        self._last_send_bitrate = float(DEFAULT_SEND_BITRATE_KBPS)
        self._last_recv_bitrate = 0.0
        
    def seed(self, num):
        np.random.seed(num)

    def reset(self):
        # self.net_env.reset_ptr()
        self.time_stamp = 0
        self.last_bit_rate = 0
        self.state = np.zeros((S_INFO, S_LEN))
        self.buffer_size = 0.
        send_bitrate_kbps = float(DEFAULT_SEND_BITRATE_KBPS)
        delay, sleep_time, self.buffer_size, rebuf, \
            frame_size, next_frame_sizes, \
            end_of_video, frame_remain, packet_loss_rate = \
            self.net_env.get_frame(send_bitrate_kbps)
        state = np.roll(self.state, -1, axis=1)

        recv_bitrate_kbps = (frame_size * 8.0) / max(delay, EPS)
        played_bitrate_kbps = send_bitrate_kbps
        fps_ratio = 60.0 / 60.0
        rtt_norm = (delay - 50.0) / 1500.0
        frame_delay_norm = (delay - 50.0) / 1000.0 / 5.0
        loss_rate = packet_loss_rate
        # send_bitrate_norm = (send_bitrate_kbps - 300.0) / (3950.0 - 300.0)
        # recv_bitrate_norm = (recv_bitrate_kbps - 300.0) / (3950.0 - 300.0)
        # played_bitrate_norm = (played_bitrate_kbps - 300.0) / (3950.0 - 300.0)
        bitrate_span = max(MAX_BITRATE_KBPS - MIN_BITRATE_KBPS, EPS)
        send_bitrate_norm = (send_bitrate_kbps - MIN_BITRATE_KBPS) / bitrate_span
        recv_bitrate_norm = (recv_bitrate_kbps - MIN_BITRATE_KBPS) / bitrate_span
        played_bitrate_norm = (played_bitrate_kbps - MIN_BITRATE_KBPS) / bitrate_span
        predict_value = recv_bitrate_norm
        predict_err = abs(send_bitrate_norm - recv_bitrate_norm)

        state[0, -1] = predict_value
        state[1, -1] = predict_err
        state[2, -1] = rtt_norm
        state[3, -1] = loss_rate
        state[4, -1] = send_bitrate_norm
        state[5, -1] = recv_bitrate_norm
        state[6, -1] = played_bitrate_norm
        state[7, -1] = fps_ratio
        state[8, -1] = frame_delay_norm
        self.state = state
        self._last_send_bitrate = send_bitrate_kbps
        self._last_recv_bitrate = recv_bitrate_kbps
        return state

    def render(self):
        return

    def step(self, action):
        # action x in [0,1] -> relative change r in [-REL_DOWN_MAX, +REL_UP_MAX]
        x = float(np.clip(action, 0.0, 1.0))
        r = -REL_DOWN_MAX + x * (REL_DOWN_MAX + REL_UP_MAX)
        send_bitrate_kbps = self._last_send_bitrate * (1.0 + r)
        send_bitrate_kbps = float(np.clip(send_bitrate_kbps, MIN_BITRATE_KBPS, MAX_BITRATE_KBPS))
        # Use true continuous bitrate in the simulator (no projection to fixed levels).
        delay, sleep_time, self.buffer_size, rebuf, \
            frame_size, next_frame_sizes, \
            end_of_video, frame_remain, packet_loss_rate = \
            self.net_env.get_frame(send_bitrate_kbps)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        smoothness = np.abs(send_bitrate_kbps - self._last_send_bitrate)
        
        # unit alignment: Mbps for bitrate/smoothness, seconds for delay, loss unitless
        send_mbps = send_bitrate_kbps / 1000.0
        smooth_mbps = smoothness / 1000.0
        delay_s = delay / 1000.0
        reward = 2 * send_mbps - 0.5 * packet_loss_rate - 0.5 * delay_s - 1.0 * smooth_mbps

        self._last_send_bitrate = send_bitrate_kbps
        kbps_levels = np.array(self.net_env.bitrate_levels_kbps, dtype=float)
        self.last_bit_rate = int(np.argmin(np.abs(kbps_levels - send_bitrate_kbps)))
        state = np.roll(self.state, -1, axis=1)

        recv_bitrate_kbps = (frame_size * 8.0) / max(delay, EPS)
        played_bitrate_kbps = send_bitrate_kbps  # approximate as sent bitrate
        fps_ratio = 60.0 / 60.0
        rtt_norm = (delay - 50.0) / 1500.0
        frame_delay_norm = (delay - 50.0) / 1000.0 / 5.0
        loss_rate = packet_loss_rate
        # send_bitrate_norm = (send_bitrate_kbps - 300.0) / (3950.0 - 300.0)
        # recv_bitrate_norm = (recv_bitrate_kbps - 300.0) / (3950.0 - 300.0)
        # played_bitrate_norm = (played_bitrate_kbps - 300.0) / (3950.0 - 300.0)
        bitrate_span = max(MAX_BITRATE_KBPS - MIN_BITRATE_KBPS, EPS)
        send_bitrate_norm = (send_bitrate_kbps - MIN_BITRATE_KBPS) / bitrate_span
        recv_bitrate_norm = (recv_bitrate_kbps - MIN_BITRATE_KBPS) / bitrate_span
        played_bitrate_norm = (played_bitrate_kbps - MIN_BITRATE_KBPS) / bitrate_span
        predict_value = recv_bitrate_norm
        predict_err = abs(send_bitrate_norm - recv_bitrate_norm)

        state[0, -1] = predict_value
        state[1, -1] = predict_err
        state[2, -1] = rtt_norm
        state[3, -1] = loss_rate
        state[4, -1] = send_bitrate_norm
        state[5, -1] = recv_bitrate_norm
        state[6, -1] = played_bitrate_norm
        state[7, -1] = fps_ratio
        state[8, -1] = frame_delay_norm

        self.state = state
        return state, reward, end_of_video, {'bitrate': send_mbps, 'packet_loss': packet_loss_rate, 'delay': delay_s}

if __name__ == "__main__":
    env = ABREnv(random_seed=666)
    state = env.reset()
    print(state)
    action = 1
    print(env.step(action=action))
