# CARLA 与数据集协作指南

本文档供负责 CARLA 环境、交通场景和风险标签的组员使用。请先阅读
“数据集的职责边界”：当前仓库可以直接启动一个随机交通 CARLA 环境，
也可以读取 comma2k19 训练未来运动模型，但还没有实现“CSV 轨迹自动回放为
CARLA NPC”的连接层。

## 1. 数据集的职责边界

| 数据来源 | 项目中的用途 | 是否能直接导入 CARLA |
| --- | --- | --- |
| comma2k19 | 训练摄像头特征、自车状态编码和未来运动预测 | 否；只有自车信息不足以重建完整交通场景 |
| NGSIM / WholeVdata2 | 提供多车辆的时间、位置、速度和轨迹 | 尚不能；需要坐标转换和 NPC 回放适配器 |
| CARLA 实时仿真 | 生成周围车辆状态、碰撞事件、MTTC、风险和严重程度标签 | 是；这是最终风险阶段的数据来源 |

推荐的整体流程是：

```text
comma2k19 ──> CNN-LSTM 自车未来运动模型 ──┐
                                          ├──> 风险融合与评估
轨迹数据 ──> CARLA NPC 交通场景 ─────────┤
CARLA ─────> 碰撞、相对状态和场景标签 ───┘
```

不要把 comma2k19 当成 CARLA 地图文件。它记录的是现实道路上的图像、
传感器数据和自车运动，不能直接生成 Highway 280 的三维道路与完整交通参与者。

## 2. 运行环境

CARLA 0.9.16 服务器建议运行在：

- Ubuntu 20.04 或 22.04，x86_64；
- NVIDIA GPU，建议至少 8 GB VRAM；
- NVIDIA 驱动、Docker、Docker Compose plugin、NVIDIA Container Toolkit；
- Python 3.10、3.11 或 3.12；
- 至少 20 GB 可用磁盘空间（不包含原始数据集）。

Apple Silicon Mac 不能原生运行当前 CARLA 服务器镜像，但可以通过浏览器访问
Ubuntu 主机上的驾驶台。

## 3. 克隆项目并启动基础 CARLA 场景

```bash
git clone https://github.com/itisyoua/honours-project-driving-risk-assessment.git
cd honours-project-driving-risk-assessment/carla_simulation

python3 doctor.py
./launch.sh --no-open
```

在同一台 Ubuntu 主机打开：

```text
http://127.0.0.1:8080
```

从另一台电脑访问时，使用：

```text
http://<Ubuntu主机IP>:8080
```

默认配置会加载 `Town03`，生成一辆自车、20 辆随机 NPC，安装 RGB 摄像头和
碰撞传感器。修改 `carla_simulation/.env` 可以调整场景：

```dotenv
CARLA_TOWN=Town05
CARLA_NPC_VEHICLES=30
CARLA_SEED=42
```

停止环境：

```bash
cd carla_simulation
./stop.sh
```

## 4. 获取并验证 comma2k19

GitHub 仓库不包含约 100 GB 的 comma2k19 原始数据。请从 `DATA.md` 中记录的
官方来源下载，并解压为：

```text
honours-project-driving-risk-assessment/
├── comma2k19/
│   ├── Chunk_1/
│   ├── Chunk_2/
│   ├── ...
│   └── Chunk_10/
└── comma2k19_data_preparation/
```

安装 Python 依赖：

```bash
cd honours-project-driving-risk-assessment
python3 -m venv comma2k19_data_preparation/.venv
comma2k19_data_preparation/.venv/bin/python -m pip install \
  -r comma2k19_data_preparation/requirements.txt
```

读取一个验证样本：

```bash
comma2k19_data_preparation/.venv/bin/python \
  -m comma2k19_data_preparation.comma2k19_dataset
```

成功时应输出一个样本的 `sequence_id`，以及以下张量维度：

```text
frames:        (30, 3, 224, 224)
state_history: (30, 8)
future_target: (20, 5)
```

模型代码应使用已经提供的 route-level `train`、`validation` 和 `test` 划分，
不能随机拆分滑动窗口，否则相邻重叠序列会造成数据泄漏。

## 5. 准备 NGSIM / WholeVdata2 轨迹

这两类轨迹比 comma2k19 更适合控制 CARLA 中的 NPC。原始文件未存入 Git，
需要根据各自的数据许可单独获取，并放入项目根目录的 `raw_data/`。

当前脚本只读取以下两个文件名：

```text
Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv
WholeVdata2.csv
```

执行：

```bash
cd honours-project-driving-risk-assessment
python3 -m venv .venv
.venv/bin/python -m pip install pandas numpy
.venv/bin/python datasetup.py
```

主要输出是：

```text
processed_data/usable_carla_ready_trajectories.csv
```

其中包含：

```text
source_file, vehicle_id, frame_id, time, x, y,
vx, vy, speed, acceleration, heading_rad, heading_deg,
lane_id, vehicle_type, nearest_vehicle_id, nearest_distance
```

注意：文件名中的 `carla_ready` 表示列已经标准化，不表示坐标已经可以直接传给
CARLA。仍需检查原始数据单位。例如 NGSIM 的部分位置和速度字段可能使用英尺制，
进入 CARLA 前必须统一转换为米、米每秒和秒。

## 6. 实现“数据轨迹 → CARLA NPC”适配器

建议新增 `carla_simulation/trajectory_replay.py`，按以下顺序开发。

### 6.1 读取和选择场景

1. 读取 `usable_carla_ready_trajectories.csv`；
2. 选择一个连续时间窗口；
3. 确定其中的自车和周围车辆；
4. 将所有车辆时间戳重采样到 CARLA tick。

当前 CARLA 环境使用固定步长 `0.05 s`（20 Hz）。若轨迹数据是 10 Hz，
需要插值到 20 Hz，或者同步修改 CARLA 的 `fixed_delta_seconds`。

### 6.2 坐标转换

数据集的二维坐标不能直接作为 CARLA 世界坐标。需要为每个场景定义：

```text
CARLA_xy = rotation(scale * dataset_xy) + translation
```

至少需要验证：

- 长度单位是否为米；
- x/y 轴方向；
- 航向角零点和顺/逆时针方向；
- 数据车道宽度是否与 CARLA 地图一致；
- 所有生成点是否落在可行驶车道附近。

最简单的第一版可以把轨迹对齐到 `Town04` 或 `Town05` 的一段直路，而不是尝试
一开始就复建现实中的 Highway 280。

### 6.3 生成和控制车辆

1. 根据 `vehicle_id` 创建车辆 actor；
2. 用车辆类型字段选择相近的 CARLA blueprint；
3. 每个 tick 更新车辆的位置和旋转；
4. 初始版本可用 `set_transform` 做确定性回放；
5. 需要真实动力学时，再改用 PID/controller 输出油门、制动和转向；
6. 场景结束后销毁所有 actor，防止重复运行时残留车辆。

### 6.4 记录风险输入与标签

每个 tick 至少记录：

- 自车与周围车辆的位置、速度、加速度和航向；
- 车辆边界框尺寸及相对距离；
- 最近前车、相对速度、TTC/MTTC；
- 交通场景 ID、天气、地图和随机种子；
- 碰撞时间、碰撞对象、碰撞冲量；
- 模型发出风险警告的时间与提前量。

碰撞概率和严重程度标签应由 CARLA 场景数据生成，不应把 comma2k19 的
`future_target` 误当作碰撞标签。

## 7. 与未来运动模型对接

模型接入 CARLA 后，应维护一个包含最近 30 帧的缓冲区。CARLA 摄像头与
自车状态构成：

```text
frames        -> [30, 3, 224, 224]
state_history -> [30, 8]
```

模型输出：

```text
future_target -> [20, 5]
```

然后把预测的自车轨迹与 CARLA 提供的 NPC 轨迹一起送入风险模块，计算未来
轨迹交叉、TTC/MTTC、碰撞概率和严重程度。

## 8. 推荐的组员分工

| 负责人 | 主要工作 | 交付物 |
| --- | --- | --- |
| 数据/模型 | comma2k19 加载、CNN-LSTM、未来轨迹评估 | 模型权重、推理接口、ADE/FDE |
| CARLA 场景 | 轨迹坐标转换、NPC 回放、传感器记录 | `trajectory_replay.py`、场景配置 |
| 风险模块 | TTC/MTTC、碰撞概率、严重程度 | 风险计算接口和评价报告 |
| 集成测试 | 固定随机种子、重复实验、结果汇总 | 可复现实验命令和测试记录 |

## 9. 完成标准

CARLA 数据集接入可以按以下检查表验收：

- [ ] Ubuntu 主机能通过 `launch.sh` 启动 CARLA；
- [ ] 其他组员能通过浏览器看到画面；
- [ ] comma2k19 单样本读取测试通过；
- [ ] 轨迹数据单位和时间频率已经确认；
- [ ] 至少一个 CSV 场景可在 CARLA 中回放；
- [ ] 同一随机种子可以重复产生一致场景；
- [ ] NPC 不会生成在道路外或互相重叠；
- [ ] 每个 tick 的自车和 NPC 状态被保存；
- [ ] 碰撞事件、TTC/MTTC 和警告提前量可以计算；
- [ ] 所有实验都记录地图、天气、场景 ID 和代码提交号。

## 10. 常见问题

### CARLA 容器无法启动

先运行：

```bash
cd carla_simulation
python3 doctor.py
docker compose logs carla
```

确认 `nvidia-smi` 正常，并且 Docker 能看到 GPU。

### 浏览器无法访问 8080

确认 `.env` 中 `WEB_HOST=0.0.0.0`，检查 Ubuntu 防火墙，并只在可信局域网内
开放端口。当前网页控制台没有用户认证，不应直接暴露到公网。

### `Comma2k19Dataset` 找不到视频

确认 `comma2k19/` 与 `comma2k19_data_preparation/` 位于同一个项目根目录，
并确认目录名称是 `Chunk_1` 至 `Chunk_10`。

### 车辆轨迹与道路不重合

不要通过不断微调单个车辆位置来修复。应统一检查单位、原点、旋转角、轴方向和
地图选段，并用同一个场景级变换处理所有车辆。
