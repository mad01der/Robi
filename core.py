import os
import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
FRAME_DURATION = 1000.0 / 30.0  # millisec, 30fps = 33.33ms per frame
TOTAL_FRAMES = 599  # frames per video sequence (from frame file length)
BUFFER_THRESH = 100.0  # millisec, RTC low latency buffer limit (3 frames)
DRAIN_BUFFER_SLEEP_TIME = 33.33  # millisec, one frame duration
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
FRAME_SIZE_FILE = './frame_size_content/frame_size_0_'
DEFAULT_MIN_BITRATE_KBPS = 80.0
DEFAULT_MAX_BITRATE_KBPS = 500.0


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED,
                 min_bitrate_kbps=DEFAULT_MIN_BITRATE_KBPS,
                 max_bitrate_kbps=DEFAULT_MAX_BITRATE_KBPS):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.frame_counter = 0
        self.buffer_size = 0
        self.last_frame_time = 0  # track timing for RTC
        self.min_bitrate_kbps = float(min_bitrate_kbps)
        self.max_bitrate_kbps = float(max_bitrate_kbps)
        if self.min_bitrate_kbps > self.max_bitrate_kbps:
            self.min_bitrate_kbps, self.max_bitrate_kbps = self.max_bitrate_kbps, self.min_bitrate_kbps
        self.bitrate_levels_kbps = self._discover_bitrate_levels(self.min_bitrate_kbps, self.max_bitrate_kbps)
        self.num_levels = len(self.bitrate_levels_kbps)

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.frame_size = {}  # in bytes
        self.frame_quality = {}  # quality scores
        loaded_profile = {}
        for bitrate_idx, bitrate in enumerate(self.bitrate_levels_kbps):
            self.frame_size[bitrate_idx] = []
            self.frame_quality[bitrate_idx] = []
            try:
                bitrate_token = str(int(round(float(bitrate))))
                with open(FRAME_SIZE_FILE + bitrate_token) as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            self.frame_size[bitrate_idx].append(float(parts[0]))
                            self.frame_quality[bitrate_idx].append(float(parts[1]))
                if len(self.frame_size[bitrate_idx]) >= TOTAL_FRAMES:
                    loaded_profile[bitrate_idx] = True
            except FileNotFoundError:
                pass

        if loaded_profile:
            loaded_indices = sorted(loaded_profile.keys())
            for bitrate_idx in range(self.num_levels):
                if bitrate_idx in loaded_profile:
                    continue
                target = self.bitrate_levels_kbps[bitrate_idx]
                nearest = min(loaded_indices, key=lambda j: abs(self.bitrate_levels_kbps[j] - target))
                src = self.bitrate_levels_kbps[nearest]
                scale = target / max(src, 1e-6)
                self.frame_size[bitrate_idx] = [v * scale for v in self.frame_size[nearest][:TOTAL_FRAMES]]
                self.frame_quality[bitrate_idx] = list(self.frame_quality[nearest][:TOTAL_FRAMES])
        else:
            # If none of the frame-size profiles are available, fall back to constants.
            for bitrate_idx in range(self.num_levels):
                self.frame_size[bitrate_idx] = [1000.0] * TOTAL_FRAMES
                self.frame_quality[bitrate_idx] = [50.0] * TOTAL_FRAMES

    def _discover_bitrate_levels(self, min_kbps, max_kbps):
        levels = []
        base_dir = os.path.dirname(FRAME_SIZE_FILE) or '.'
        prefix = os.path.basename(FRAME_SIZE_FILE)
        try:
            for name in os.listdir(base_dir):
                if not name.startswith(prefix):
                    continue
                suffix = name[len(prefix):]
                if not suffix.isdigit():
                    continue
                kbps = float(int(suffix))
                if min_kbps <= kbps <= max_kbps:
                    levels.append(kbps)
        except FileNotFoundError:
            levels = []

        levels = sorted(set(levels))
        if not levels:
            levels = [float(min_kbps), float(max_kbps)]
        else:
            if levels[0] > min_kbps:
                levels.insert(0, float(min_kbps))
            if levels[-1] < max_kbps:
                levels.append(float(max_kbps))
        return levels
    def reset_to_trace(self, trace_idx):
        self.trace_idx = int(trace_idx) % len(self.all_cooked_time)
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = 0
        self.frame_counter = 0
        self.buffer_size = 0

    def _resolve_bitrate_kbps(self, bitrate_or_level):
        """
        Accept either:
        - discrete level index (legacy path), or
        - continuous bitrate in kbps (new path).
        """
        if isinstance(bitrate_or_level, (int, np.integer)) and 0 <= int(bitrate_or_level) < self.num_levels:
            return float(self.bitrate_levels_kbps[int(bitrate_or_level)])
        return float(bitrate_or_level)

    def _interp_frame_metrics(self, bitrate_kbps, frame_idx):
        levels = np.array(self.bitrate_levels_kbps, dtype=float)
        if len(levels) == 1:
            return self.frame_size[0][frame_idx], self.frame_quality[0][frame_idx]
        bitrate_kbps = float(np.clip(bitrate_kbps, levels[0], levels[-1]))

        # Boundary cases: clip to nearest endpoint level.
        if bitrate_kbps <= levels[0]:
            return self.frame_size[0][frame_idx], self.frame_quality[0][frame_idx]
        if bitrate_kbps >= levels[-1]:
            last = len(levels) - 1
            return self.frame_size[last][frame_idx], self.frame_quality[last][frame_idx]

        hi = int(np.searchsorted(levels, bitrate_kbps, side='right'))
        lo = hi - 1
        lo_kbps = levels[lo]
        hi_kbps = levels[hi]
        w = (bitrate_kbps - lo_kbps) / (hi_kbps - lo_kbps)

        frame_size = (1.0 - w) * self.frame_size[lo][frame_idx] + w * self.frame_size[hi][frame_idx]
        frame_quality = (1.0 - w) * self.frame_quality[lo][frame_idx] + w * self.frame_quality[hi][frame_idx]
        return frame_size, frame_quality

    def get_frame(self, bitrate_or_level):
        bitrate_kbps = self._resolve_bitrate_kbps(bitrate_or_level)

        frame_size, _frame_quality = self._interp_frame_metrics(bitrate_kbps, self.frame_counter)
        
        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        frame_counter_sent = 0  # in bytes
        
        while True:  # download frame over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time
	    
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if frame_counter_sent + packet_payload > frame_size:

                fractional_time = (frame_size - frame_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            frame_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

	    # add a multiplicative noise to the delay
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # For RTC, calculate packet loss rate based on network conditions
        # Simple model: higher delay and lower throughput = higher packet loss
        current_throughput = self.cooked_bw[self.mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE
        # 缩小当网络带宽低时的固有虚假丢包率，以便GCC模型可以更顺利尝试提速升档 (<0.02)
        # 将静态底端延迟从 50ms 扣减提升到 100ms (抵消 80ms 的 LINK_RTT)
        packet_loss_rate = max(0.0, (delay - 100) / 1000 + (1000000 - current_throughput) / 100000000)

        # rebuffer time (for RTC, this represents frame drops)
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new frame
        self.buffer_size += FRAME_DURATION

        # sleep if buffer gets too large (less aggressive for RTC)
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        return_buffer_size = self.buffer_size

        self.frame_counter += 1
        frame_remain = TOTAL_FRAMES - self.frame_counter

        end_of_video = False
        if self.frame_counter >= TOTAL_FRAMES:
            end_of_video = True
            self.buffer_size = 0
            self.frame_counter = 0

            # pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the trace
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_frame_size = []
        for i in range(self.num_levels):
            next_frame_size.append(self.frame_size[i][self.frame_counter])

        return delay, \
               sleep_time, \
               return_buffer_size / MILLISECONDS_IN_SECOND, \
               rebuf / MILLISECONDS_IN_SECOND, \
               frame_size, \
               next_frame_size, \
               end_of_video, \
               frame_remain, \
               packet_loss_rate
            
