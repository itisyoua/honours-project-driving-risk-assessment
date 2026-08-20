#!/usr/bin/env python3
"""A small browser-controlled CARLA environment for CARLA 0.9.16."""

from __future__ import annotations

import argparse
import atexit
import io
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from PIL import Image

try:
    import carla
except ImportError as exc:
    raise SystemExit(
        "CARLA Python API is missing. Use Python 3.10-3.12 on Linux x86_64 and run: "
        "python -m pip install -r requirements.txt"
    ) from exc


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CARLA 驾驶台</title>
<style>
:root{color-scheme:dark;--panel:#10151dcc;--accent:#4ee1a0;--muted:#9ba7b4}
*{box-sizing:border-box}body{margin:0;background:#080b10;color:#eef3f7;font:15px system-ui,sans-serif;overflow:hidden}
#view{width:100vw;height:100vh;object-fit:contain;display:block;background:#000}
.hud{position:fixed;inset:18px auto auto 18px;background:var(--panel);padding:14px 16px;border:1px solid #ffffff22;border-radius:12px;backdrop-filter:blur(8px);min-width:240px}
.title{font-size:18px;font-weight:700;margin-bottom:8px}.accent{color:var(--accent)}.muted{color:var(--muted)}
.row{display:flex;justify-content:space-between;gap:20px;line-height:1.7}.help{position:fixed;right:18px;bottom:18px;background:var(--panel);padding:12px 15px;border-radius:10px;color:var(--muted)}
kbd{color:#fff;border:1px solid #647180;border-radius:4px;padding:1px 5px;background:#252d38}.offline{color:#ff7b72}
</style></head>
<body><img id="view" src="/stream.mjpg" alt="CARLA RGB camera stream">
<section class="hud"><div class="title">CARLA <span class="accent">0.9.16</span></div>
<div class="row"><span>连接</span><span id="connection">正在连接…</span></div>
<div class="row"><span>模式</span><span id="mode">自动驾驶</span></div>
<div class="row"><span>速度</span><span id="speed">0 km/h</span></div>
<div class="row"><span>地图</span><span id="map">—</span></div>
<div class="row"><span>碰撞</span><span id="collisions">0</span></div></section>
<div class="help"><kbd>WASD</kbd>/<kbd>方向键</kbd> 驾驶　<kbd>空格</kbd> 刹车　<kbd>P</kbd> 切换自动驾驶　<kbd>R</kbd> 重生</div>
<script>
const keys=new Set();let autopilot=true;
async function send(extra={}){const t=keys.has('w')||keys.has('arrowup')?1:0,b=keys.has('s')||keys.has('arrowdown')||keys.has(' ')?1:0;
 const steer=(keys.has('a')||keys.has('arrowleft')?-1:0)+(keys.has('d')||keys.has('arrowright')?1:0);
 try{await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({throttle:t,brake:b,steer,...extra})})}catch(e){}}
addEventListener('keydown',e=>{const k=e.key.toLowerCase();if([' ','arrowup','arrowdown','arrowleft','arrowright'].includes(k))e.preventDefault();if(e.repeat)return;keys.add(k);if(k==='p'){autopilot=!autopilot;send({autopilot})}if(k==='r')send({respawn:true})});
addEventListener('keyup',e=>keys.delete(e.key.toLowerCase()));setInterval(()=>{if(!autopilot)send()},50);
setInterval(async()=>{try{const s=await (await fetch('/api/status')).json();document.querySelector('#connection').textContent='已连接';document.querySelector('#connection').className='accent';document.querySelector('#mode').textContent=s.autopilot?'自动驾驶':'手动驾驶';autopilot=s.autopilot;document.querySelector('#speed').textContent=s.speed_kmh.toFixed(1)+' km/h';document.querySelector('#map').textContent=s.map;document.querySelector('#collisions').textContent=s.collisions}catch(e){const x=document.querySelector('#connection');x.textContent='已断开';x.className='offline'}},500);
</script></body></html>"""


@dataclass
class SharedState:
    frame: bytes | None = None
    frame_id: int = -1
    autopilot: bool = True
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    speed_kmh: float = 0.0
    map_name: str = "—"
    collisions: int = 0
    respawn: bool = False
    ready: bool = False
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    condition: threading.Condition = field(default_factory=threading.Condition)


class CarlaSession:
    def __init__(self, args: argparse.Namespace, state: SharedState):
        self.args = args
        self.state = state
        self.client: Any = None
        self.world: Any = None
        self.tm: Any = None
        self.original_settings: Any = None
        self.ego: Any = None
        self.camera: Any = None
        self.actors: list[Any] = []
        self.running = True
        self.rng = random.Random(args.seed)
        self.last_autopilot = True

    def connect(self) -> None:
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world(self.args.town)
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        self.tm = self.client.get_trafficmanager(self.args.tm_port)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(self.args.seed)
        self.tm.global_percentage_speed_difference(12.0)
        with self.state.lock:
            self.state.map_name = self.world.get_map().name.split("/")[-1]
        self.spawn_scene()

    def spawn_scene(self) -> None:
        blueprints = self.world.get_blueprint_library()
        points = self.world.get_map().get_spawn_points()
        self.rng.shuffle(points)
        preferred = blueprints.filter("vehicle.tesla.model3")
        ego_bp = preferred[0] if preferred else blueprints.filter("vehicle.*")[0]
        ego_bp.set_attribute("role_name", "hero")
        for point in points:
            self.ego = self.world.try_spawn_actor(ego_bp, point)
            if self.ego:
                break
        if not self.ego:
            raise RuntimeError("No free spawn point for the ego vehicle")
        self.actors.append(self.ego)
        self.ego.set_autopilot(True, self.args.tm_port)

        vehicle_bps = [bp for bp in blueprints.filter("vehicle.*") if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) == 4]
        occupied = self.ego.get_location()
        candidates = [p for p in points if p.location.distance(occupied) > 8.0]
        for point in candidates[: self.args.vehicles]:
            bp = self.rng.choice(vehicle_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", self.rng.choice(bp.get_attribute("color").recommended_values))
            actor = self.world.try_spawn_actor(bp, point)
            if actor:
                actor.set_autopilot(True, self.args.tm_port)
                self.actors.append(actor)

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.args.width))
        camera_bp.set_attribute("image_size_y", str(self.args.height))
        camera_bp.set_attribute("fov", "100")
        camera_bp.set_attribute("sensor_tick", "0.05")
        camera_tf = carla.Transform(carla.Location(x=0.8, z=1.65))
        self.camera = self.world.spawn_actor(camera_bp, camera_tf, attach_to=self.ego)
        self.actors.append(self.camera)
        self.camera.listen(self.on_image)

        collision_bp = blueprints.find("sensor.other.collision")
        collision = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego)
        collision.listen(self.on_collision)
        self.actors.append(collision)

    def on_image(self, image: Any) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb = array[:, :, :3][:, :, ::-1]
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="JPEG", quality=82)
        with self.state.condition:
            self.state.frame = buffer.getvalue()
            self.state.frame_id = image.frame
            self.state.condition.notify_all()

    def on_collision(self, _event: Any) -> None:
        with self.state.lock:
            self.state.collisions += 1

    def update(self) -> None:
        with self.state.lock:
            autopilot = self.state.autopilot
            control = (self.state.throttle, self.state.steer, self.state.brake)
            respawn = self.state.respawn
            self.state.respawn = False
        if autopilot != self.last_autopilot:
            self.ego.set_autopilot(autopilot, self.args.tm_port)
            self.last_autopilot = autopilot
        if respawn:
            points = self.world.get_map().get_spawn_points()
            self.ego.set_transform(self.rng.choice(points))
            self.ego.set_target_velocity(carla.Vector3D())
            self.ego.set_target_angular_velocity(carla.Vector3D())
        if not autopilot:
            self.ego.apply_control(carla.VehicleControl(throttle=control[0], steer=control[1], brake=control[2]))
        velocity = self.ego.get_velocity()
        speed = 3.6 * (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
        transform = self.ego.get_transform()
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(transform.transform(carla.Location(x=-7.0, z=4.0)), transform.rotation))
        with self.state.lock:
            self.state.speed_kmh = speed

    def run(self) -> None:
        try:
            self.connect()
            with self.state.lock:
                self.state.ready = True
            while self.running:
                self.world.tick()
                self.update()
        except Exception as exc:  # surfaced through the status endpoint and stderr
            with self.state.lock:
                self.state.error = str(exc)
            print(f"CARLA session failed: {exc}", flush=True)
        finally:
            self.close()

    def close(self) -> None:
        self.running = False
        if self.camera:
            self.camera.stop()
        for actor in reversed(self.actors):
            try:
                actor.destroy()
            except RuntimeError:
                pass
        self.actors.clear()
        if self.tm:
            self.tm.set_synchronous_mode(False)
        if self.world and self.original_settings:
            self.world.apply_settings(self.original_settings)


def create_app(args: argparse.Namespace) -> tuple[Flask, CarlaSession]:
    app = Flask(__name__)
    state = SharedState()
    session = CarlaSession(args, state)

    @app.get("/")
    def index() -> str:
        return render_template_string(INDEX_HTML)

    @app.get("/api/status")
    def status() -> Response:
        with state.lock:
            payload = {
                "ready": state.ready,
                "autopilot": state.autopilot,
                "speed_kmh": state.speed_kmh,
                "map": state.map_name,
                "collisions": state.collisions,
                "error": state.error,
            }
        return jsonify(payload)

    @app.post("/api/control")
    def control() -> Response:
        data = request.get_json(silent=True) or {}
        with state.lock:
            if "autopilot" in data:
                state.autopilot = bool(data["autopilot"])
            state.throttle = min(1.0, max(0.0, float(data.get("throttle", state.throttle))))
            state.steer = min(1.0, max(-1.0, float(data.get("steer", state.steer))))
            state.brake = min(1.0, max(0.0, float(data.get("brake", state.brake))))
            state.respawn = state.respawn or bool(data.get("respawn", False))
        return jsonify(ok=True)

    @app.get("/stream.mjpg")
    def stream() -> Response:
        def frames():
            last_id = -1
            while True:
                with state.condition:
                    state.condition.wait_for(lambda: state.frame_id != last_id or state.error, timeout=2.0)
                    if state.frame is None:
                        if state.error:
                            return
                        continue
                    frame, last_id = state.frame, state.frame_id
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    thread = threading.Thread(target=session.run, name="carla-session", daemon=True)
    thread.start()
    return app, session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("CARLA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CARLA_PORT", "2000")))
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--web-host", default=os.getenv("WEB_HOST", "0.0.0.0"))
    parser.add_argument("--web-port", type=int, default=int(os.getenv("WEB_PORT", "8080")))
    parser.add_argument("--town", default=os.getenv("CARLA_TOWN", "Town03"))
    parser.add_argument("--vehicles", type=int, default=int(os.getenv("CARLA_NPC_VEHICLES", "20")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("CARLA_SEED", "42")))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app, session = create_app(args)
    atexit.register(session.close)
    print(f"Browser UI: http://127.0.0.1:{args.web_port}", flush=True)
    app.run(host=args.web_host, port=args.web_port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
