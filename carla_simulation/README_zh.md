# CARLA 0.9.16 可直接启动环境

这是一个小型但完整的 CARLA 环境：启动 CARLA 服务器、生成自车和 NPC 交通，并在浏览器里显示第一人称 RGB 相机。浏览器支持自动驾驶与键盘驾驶，因此不需要把 Unreal Engine 窗口传到另一台电脑。

## 先说明这台 Mac 的限制

当前电脑是 Apple M1 Pro / macOS / arm64。CARLA 0.9.16 官方服务器、Docker 镜像和 Python wheel 只发布给 Ubuntu/Windows x86_64；macOS 不能原生运行这套环境。Mac 可以作为浏览器控制端，CARLA 服务器需要运行在另一台符合以下条件的机器：

- Ubuntu 20.04 或 22.04，x86_64；
- NVIDIA RTX 2070 或更高，建议至少 8 GB VRAM；
- NVIDIA 驱动、Docker、Docker Compose plugin、NVIDIA Container Toolkit；
- Python 3.10、3.11 或 3.12；
- 至少约 20 GB 空闲磁盘；
- 局域网允许 TCP 端口 8080（浏览器）、2000/2001（仅需要直接访问 CARLA API 时开放）。

## Ubuntu 上一键启动

把整个 `carla_simulation` 目录复制到 Ubuntu 主机，然后：

```bash
cd carla_simulation
chmod +x launch.sh stop.sh
./launch.sh
```

首次启动会下载约 8 GB 的压缩 Docker 镜像并创建 Python 虚拟环境，解压后总体空间更大。完成后会打开：

```text
http://127.0.0.1:8080
```

停止时在启动终端按 `Ctrl+C`。异常退出后可运行 `./stop.sh`。

## 在这台 Mac 上打开

1. Ubuntu 服务器上运行 `./launch.sh --no-open`。
2. 将 `.env.example` 复制成 `.env`。
3. 把 `.env` 中的地址改成 Ubuntu 主机地址，例如：

```dotenv
CARLA_WEB_URL=http://192.168.1.50:8080
```

4. 在 Mac 上双击 `open_carla.command`，或用浏览器打开上面的地址。

只应在可信局域网内开放 8080；这个轻量控制台没有登录认证，不要直接暴露到公网。

## 操作

- `P`：自动驾驶 / 手动驾驶切换；
- `W A S D` 或方向键：油门、转向与制动；
- 空格：制动；
- `R`：把自车移动到新的出生点。

默认场景是 `Town03`、20 辆 NPC、自车自动驾驶。可修改 `.env`：

```dotenv
CARLA_TOWN=Town05
CARLA_NPC_VEHICLES=30
CARLA_SEED=42
```
## 自检与故障排查

```bash
python3 doctor.py
docker compose logs carla
```

常见问题：

- `nvidia-smi not found`：先安装正确的 NVIDIA 驱动；
- Docker 看不到 GPU：安装 NVIDIA Container Toolkit，并验证 `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`；
- 浏览器无画面：等待地图和 shader 完成首次加载，再查看 `docker compose logs carla`；
- 远程打不开 8080：检查 Ubuntu 防火墙和 `.env` 内的 IP；
- 端口被占用：停止原有 CARLA/网页进程，或同步修改 compose、`.env` 和启动参数。

## 文件说明

- `compose.yaml`：官方 CARLA 0.9.16 GPU 容器；
- `web_simulator.py`：场景、交通流、传感器与网页驾驶台；
- `launch.sh`：创建环境、启动服务、打开网页；
- `doctor.py`：只读环境诊断；
- `open_carla.command`：Mac 双击打开远程驾驶台。

