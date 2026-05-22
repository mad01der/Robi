# ATLA 项目训练流程交接文档

本文档说明这个项目的训练主流程、各模块职责、关键参数与取值范围、以及落地运行时的注意事项。

## 1. 项目目标与核心思路

该项目是一个用于实时视频码率控制（ABR）的强化学习系统，核心是：

- `ATLA_agent`：主策略网络（Actor-Critic + PPO），输出连续动作。
- `ATLA_adv`：对抗扰动网络，在状态输入上添加扰动，提高主策略鲁棒性。
- `env/core/load_trace`：网络轨迹仿真环境，返回延迟、丢包、吞吐等反馈。

训练采用双阶段交替：

1. 先更新 `agent`（在扰动状态上做 PPO）。
2. 再冻结 `agent`，更新 `adversary`（让扰动更“难”）。

---

## 2. 代码结构与职责

训练主链路相关文件如下：

- `ATLA_train.py`
  - 主训练入口（推荐默认训练脚本）。
  - 负责 epoch 循环、采样、训练、保存模型、触发测试、写 TensorBoard。
- `ATLA_agent.py`
  - 主策略网络与价值网络定义。
  - 包含 Beta 分布策略、PPO/Dual-PPO 损失、价值回报计算。
- `ATLA_adv.py`
  - 对抗扰动网络定义与训练逻辑。
  - 通过添加有界扰动来构造困难样本。
- `env.py`
  - 对环境 `core.Environment` 的包装。
  - 定义状态特征、动作到码率增量的映射、奖励函数。
- `core.py`
  - 底层网络仿真器（按带宽轨迹下载帧，生成 delay、loss、buffer 等）。
- `load_trace.py`
  - 加载训练/测试带宽轨迹文件。
- `atla_test.py`
  - 给定模型与测试轨迹，逐 trace 评估并落日志。

常用变体脚本：

- `ATLA_train_noadv.py`：无对抗扰动版本（仅训练 agent）。
- `ATLA_train_nopomdp.py`：去时序版本（把状态列复制成“单时刻展开”）。

---

## 3. 训练完整流程（以 `ATLA_train.py` 为准）

## 3.1 初始化阶段

1. 设置随机种子与 PyTorch 性能选项（线程数、TF32、cudnn benchmark）。
2. 初始化环境：`env = ABREnv(0)`。
3. 初始化网络：
   - `adversary = ATLA_adv.Network(...)`
   - `agent = ATLA_agent.Network(...)`
4. 将 `agent/adversary` 的 `PPO_TRAINING_EPO` 从默认 `5` 调为 `3`（加速单轮训练）。
5. 尝试恢复 checkpoint：
   - 优先加载 `*_latest.pth`
   - 再从 `*_ep_<N>.pth` 中找最大 epoch 恢复。
6. 初始化 TensorBoard writer 和测试日志文件。

## 3.2 每个 epoch 的两阶段训练

### Phase 1：训练主策略 `agent`

1. `env.reset()` 后滚动采样最多 `TRAIN_SEQ_LEN` 步。
2. 每步流程：
   - `agent.predict(state)` 输出 Beta 分布参数 `alpha, beta`
   - 采样动作 `x ~ Beta(alpha, beta)`，`x in (0,1)`
   - `env.step(x)` 获得下一状态和奖励
   - 缓存 `s_batch/a_batch/p_batch/r_batch`
3. 用 `agent.compute_v(...)` 计算回报目标 `v_batch`。
4. 用 `adversary.predict(s_batch)` 生成扰动状态 `perturbed_obs`。
5. `agent.train(perturbed_obs, ...)` 更新策略与价值网络。
6. 周期性保存 `agent`，并调用 `atla_test.py` 跑测试轨迹，记录奖励统计。

### Phase 2：训练对抗网络 `adversary`

1. 重新采样一批轨迹数据（同样来自 agent 与 env 交互）。
2. 冻结 `agent` 参数（`requires_grad = False`）。
3. `adversary.train(...)`：
   - 先对状态加扰动
   - 再通过 `agent.predict_adv(...)` 得到策略输出和值函数
   - 以“反向目标”更新对抗网络（让 agent 在扰动下更难优化）
4. 周期性保存 `adversary`。
5. 每轮都额外覆盖保存 `agent_nn_model_latest.pth` 与 `adv_nn_model_latest.pth`。

---

## 4. 状态、动作、奖励设计

## 4.1 状态定义（`S_DIM = [9,8]`）

状态是 `9 x 8` 矩阵，按时间滚动，最后一列是最新观测：

- `state[0]`：`predict_value`（当前用接收码率归一化值替代）
- `state[1]`：`predict_err = |send_norm - recv_norm|`
- `state[2]`：`rtt_norm = (delay - 50) / 1500`
- `state[3]`：`loss_rate`（0~1）
- `state[4]`：`send_bitrate_norm`
- `state[5]`：`recv_bitrate_norm`
- `state[6]`：`played_bitrate_norm`
- `state[7]`：`fps_ratio`（当前固定 `60/60=1`）
- `state[8]`：`frame_delay_norm = (delay - 50)/1000/5`

码率归一化采用 `env.py` 的上下限配置：

- `norm = (kbps - MIN_BITRATE_KBPS) / (MAX_BITRATE_KBPS - MIN_BITRATE_KBPS)`
- 当前默认区间：`80 ~ 500 kbps`（可改）。

## 4.2 动作定义（连续动作）

策略输出通过 Beta 分布采样得到 `x in (0,1)`，环境内映射为相对码率变化：

- `r = -REL_DOWN_MAX + x * (REL_DOWN_MAX + REL_UP_MAX)`（默认 `REL_DOWN_MAX=0.5`，`REL_UP_MAX=0.2`）
- 即：`x=0 -> r=-50%`，`x=1 -> r=+20%`（降快升慢）
- 新发送码率 = 上次发送码率 × `(1 + r)`
- 最终裁剪到 `[MIN_BITRATE_KBPS, MAX_BITRATE_KBPS]` 后，直接按连续 `kbps` 驱动仿真。
- `core.py` 会在可用锚点（`frame_size_0_*`）之间做线性插值，用于近似连续码率内容大小。

## 4.3 奖励函数（`env.py`）

奖励为：

- `reward = 2*send_mbps - 0.5*packet_loss_rate - 0.5*delay_s - 1.0*smooth_mbps`

含义：

- 提升码率（正向）
- 惩罚丢包、时延、码率波动（负向）

---

## 5. 关键预设参数与作用

## 5.1 训练脚本参数（`ATLA_train.py`）

- `S_DIM = [9, 8]`：状态维度（9个特征，8步历史）。
- `A_DIM = 8`：动作维度占位（当前实际动作是 1 维连续值，8 主要用于熵初值）。
- `ACTOR_LR_RATE = 1e-4`：agent/adv Adam 学习率。
- `TRAIN_SEQ_LEN = 500`：每个 phase 最长采样步数。
- `TRAIN_EPOCH = 500000`：总训练 epoch。
- `MODEL_SAVE_INTERVAL = 30`：每隔多少 epoch 保存并测试。
- `RANDOM_SEED = 42`：随机种子。
- `SUMMARY_DIR` / `LOG_ROOT`：模型与 TensorBoard 主目录（`../../tf-logs/`）。
- `TEST_LOG_FOLDER`：测试日志输出目录（`../../tf-logs/test_results/`）。
- `ADV_NN_MODEL` / `AGENT_NN_MODEL`：手动指定预训练模型路径（默认 `None`）。

## 5.2 Agent 参数（`ATLA_agent.py`）

- `FEATURE_NUM = 128`：每路特征抽取层宽度。
- `GAMMA = 0.99`：回报折扣因子。
- `EPS = 0.2`：PPO clip 范围（`[1-EPS, 1+EPS]`）。
- `PPO_TRAINING_EPO`：每批数据做几轮优化，默认 `5`，主脚本中改为 `3`。
- `_entropy_weight = log(action_dim)`：熵正则初值（探索强度）。
- `H_target = 0.1`：目标熵，用于自适应调节熵权重。

## 5.3 对抗网络参数（`ATLA_adv.py`）

- `perturbation_bound = 0.01`：单次扰动幅度上界（通过 `tanh` 后映射到 `[0, 0.02]` 附近增量）。
- `PPO_TRAINING_EPO`、`EPS`、`H_target`、`_entropy_weight`：
  - 与 agent 类似，但优化目标是“对抗地”恶化 agent 的训练样本。

## 5.4 环境与网络仿真参数（`env.py` / `core.py`）

- `MIN_BITRATE_KBPS` / `MAX_BITRATE_KBPS`
  - 码率控制上下限（训练主配置），默认 `80/500`。
- `DEFAULT_SEND_BITRATE_KBPS`
  - `reset` 后初始发送码率，默认 `300`。
- `bitrate_levels_kbps`（`core.Environment` 运行时生成）
  - 自动扫描 `frame_size_content/frame_size_0_*` 得到区间内可用锚点；
  - 若上下限不在现有文件中，会补上边界锚点并通过邻近档位缩放构造帧大小序列。
- `TOTAL_FRAMES = 599`
  - 一段视频序列帧数，上限到达则 `end_of_video=True` 并切换 trace。
- `FRAME_DURATION = 33.33ms`（30fps）
- `BUFFER_THRESH = 100ms`
  - RTC 低时延缓冲阈值。
- `LINK_RTT = 80ms`
  - 仿真链路基础 RTT。
- `NOISE_LOW/HIGH = 0.9/1.1`
  - 传输时延乘性噪声范围。
- `PACKET_PAYLOAD_PORTION = 0.95`
  - 有效载荷比例。
- 丢包估计：
  - `packet_loss_rate = max(0, (delay-100)/1000 + (1e6-throughput)/1e8)`
  - 延迟越高、吞吐越低时，估计丢包越高。

---

## 6. 训练数据与文件依赖

- 轨迹加载默认目录：`load_trace.py` 中 `./train/huawei500kbps/train/`。
- `ATLA_train.py` 测试目录：`./train/huawei500kbps/test/`
- 帧大小文件目录（由 `core.py` 读取）：
  - `./frame_size_content/frame_size_0_<bitrate>`
  - 不再要求手动维护固定离散列表；只要目录中存在若干可用 `<bitrate>` 文件，`core.py` 就会在 `[MIN, MAX]` 内自动发现锚点并插值。

如果交接后目录结构变化，优先改这几个路径常量。

---

## 7. 训练产物与日志说明

- 模型文件：
  - `agent_nn_model_ep_<epoch>.pth`
  - `adv_nn_model_ep_<epoch>.pth`
  - `agent_nn_model_latest.pth`
  - `adv_nn_model_latest.pth`
- 测试日志：
  - `LOG_FILE + '_test.txt'` 中记录每轮测试分位数统计（min/5%/mean/50%/95%/max）。
- TensorBoard 标量：
  - `Entropy Weight`
  - `Reward`
  - `Entropy`

---

## 8. 运行与交接建议

最低建议交接步骤：

1. 先确认 `frame_size_content` 与训练/测试 trace 路径可读。
2. 从 `*_latest.pth` 断点恢复跑几个 epoch，确认 loss/reward 正常变化。
3. 用 `atla_test.py` 对固定测试集出日志，和历史结果做对比。
4. 打开 TensorBoard 看 `Reward/Entropy` 是否稳定。

---

## 9. 当前代码中的注意点（建议接手后优先核对）

- `ATLA_train.py` 的 Phase 2 采样段中，`a_batch/r_batch/p_batch` 的追加位置缩进看起来在循环外，可能导致只记录最后一步样本；建议接手后先核实这里是否为预期。
- `A_DIM=8` 与实际连续动作维度（1 维）不是同一个概念，当前主要用于熵权重初值，不影响动作输出形态。
- `PACKET_LOSS_PENALTY/DELAY_PENALTY/SMOOTH_PENALTY` 在 `env.py` 定义了但当前 reward 未直接使用这些常量（使用的是固定系数表达式）。
- 当前 `frame_size_content` 下若只有极少锚点文件（例如仅有 `frame_size_0_300`），`core.py` 会用邻近档位缩放补齐序列；可运行，但建议补充更多锚点文件以提升仿真真实性。
---

## 10. 一句话给接手人

先把 `ATLA_train.py -> env.py -> ATLA_agent.py -> ATLA_adv.py -> core.py` 这 5 个文件串起来看；你会发现这是一个“连续动作 Beta-PPO + 对抗扰动”的双阶段 ABR 训练器，最关键的是状态归一化范围、动作到码率映射、reward 系数、以及路径配置是否与当前数据集匹配。
