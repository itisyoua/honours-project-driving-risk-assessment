# CARLA 0.9.16 Ready-to-Run Environment

This directory provides a small but complete CARLA environment. It starts the
CARLA server, spawns an ego vehicle and NPC traffic, and displays a first-person
RGB camera in a browser. The dashboard supports both autopilot and keyboard
driving, so the Unreal Engine window does not need to be forwarded to another
computer.

## Mac limitations

The current development computer is an Apple M1 Pro running macOS on arm64.
The official CARLA 0.9.16 server, Docker image, and Python wheel are released
for Ubuntu/Windows x86_64 and cannot run natively on this Mac. The Mac can be
used as a browser client, while the CARLA server runs on a machine with:

- Ubuntu 20.04 or 22.04 on x86_64;
- an NVIDIA RTX 2070 or newer, with at least 8 GB VRAM recommended;
- an NVIDIA driver, Docker, the Docker Compose plugin, and NVIDIA Container Toolkit;
- Python 3.10, 3.11, or 3.12;
- approximately 20 GB or more of free disk space;
- TCP port 8080 available on the local network for the browser, with ports
  2000/2001 required only for direct CARLA API access.

## One-command Ubuntu startup

Copy or clone the project onto the Ubuntu host, then run:

```bash
cd carla_simulation
chmod +x launch.sh stop.sh
./launch.sh
```

The first run downloads an approximately 8 GB compressed Docker image and
creates a Python virtual environment. The extracted image requires additional
disk space. When startup is complete, open:

```text
http://127.0.0.1:8080
```

Press `Ctrl+C` in the launch terminal to stop the environment. If it exits
unexpectedly, run `./stop.sh`.

## Open the environment from the Mac

1. Run `./launch.sh --no-open` on the Ubuntu server.
2. Copy `.env.example` to `.env`.
3. Set the Ubuntu host address in `.env`, for example:

```dotenv
CARLA_WEB_URL=http://192.168.1.50:8080
```

4. Double-click `open_carla.command` on the Mac, or open the address above in a
   browser.

Expose port 8080 only on a trusted local network. This lightweight dashboard
does not provide authentication and must not be exposed directly to the public
internet.

## Controls

- `P`: toggle autopilot/manual driving;
- `W A S D` or arrow keys: throttle, steering, and braking;
- Space: brake;
- `R`: move the ego vehicle to a new spawn point.

The default scene is `Town03` with 20 NPCs and the ego vehicle on autopilot.
Edit `.env` to change the scene:

```dotenv
CARLA_TOWN=Town05
CARLA_NPC_VEHICLES=30
CARLA_SEED=42
```

## Diagnostics and troubleshooting

```bash
python3 doctor.py
docker compose logs carla
```

Common issues:

- `nvidia-smi not found`: install the correct NVIDIA driver.
- Docker cannot see the GPU: install NVIDIA Container Toolkit and verify with
  `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- The browser shows no image: allow the map and shaders to finish loading, then
  inspect `docker compose logs carla`.
- Port 8080 is unavailable remotely: check the Ubuntu firewall and the address
  configured in `.env`.
- A port is already in use: stop existing CARLA or dashboard processes, or
  update the port consistently in Compose, `.env`, and the launch arguments.

## Files

- `compose.yaml`: official CARLA 0.9.16 GPU container.
- `web_simulator.py`: scene, traffic, sensors, and browser dashboard.
- `launch.sh`: creates the environment, starts the services, and opens the dashboard.
- `doctor.py`: read-only environment diagnostics.
- `open_carla.command`: opens the remote dashboard from macOS.
