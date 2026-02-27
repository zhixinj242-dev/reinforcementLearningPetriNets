"""
visual.py
visual.py = 统一的可视化入口
所有可视化都在这里：
✓ Baseline V1/V2（固定规则）
✓ Agent（CDQN/DQN 训练的模型）
✓ Imitation（GAIL 模仿学习）
# Baseline V1
python visual.py --mode baseline --baseline-version v1 --render frames --out frames_b1 --steps 50

# Baseline V2
python visual.py --mode baseline --baseline-version v2 --render frames --out frames_b2 --steps 50

# CDQN 模型
python visual.py --mode agent --path lido-run-events/agent_s1.5c0.0w1.0mw1.5t0.0_cdqn.pt --render frames --out frames_cdqn --steps 50

# DQN 模型
python visual.py --mode agent --path lido-run-events/agent_s1.5c0.0w1.0mw1.5t0.0_dqn.pt --render frames --out frames_dqn --steps 50

# GAIL 模仿学习
python visual.py --mode imitation --path gail_policy.pt --render frames --out frames_gail --steps 50

【目标】
把“Baseline / 训练后的智能体”如何指挥红绿灯、以及它如何与环境（env）和 Petri 网配合，
用“可运行 + 可视化 + 大量中文注释”的方式讲清楚。

【这份脚本回答的问题】
1) baseline / 智能体到底输出什么？—— 输出的是一个“动作 action（离散整数）”
2) action 在 env 里怎么用？—— env.step(action) 会把 action 映射成 Petri 网的“变迁 transition”，尝试 fire
3) Petri 网如何决定哪个方向绿灯？—— token 在哪些 place 上，就表示哪些相位（place）是激活的；车道绑定 place
4) 车怎么动？—— env._do_driving()：绿灯车道按泊松分布放行；所有车等待时间+1，并按泊松分布来新车

【重要约束（非常关键）】
项目里的约束DQN（`agents/constrained_dqn.py` 的 `CDQN.act`）期望输入是一个
“扁平化后的状态向量 torch.Tensor”，而不是 dict 观测。
所以本脚本会把 env 的 dict 观测按固定顺序 flatten 成 1D 向量，再喂给智能体。

【渲染模式】
- pygame: 需要有桌面/显示器（本地Windows一般OK）
- frames: 适合无桌面服务器，使用 SDL dummy 驱动“离屏渲染”，逐帧保存 PNG 到输出目录

【文件角色】（本仓库新增）：
- 提供 Baseline / DQN 智能体 / GAIL 策略三种控制源的可视化。
- 与 env 配合方式：controller.act(obs,t) 得到 action → env.step(action) 执行；红绿灯由 Petri 网 token 决定，车辆由 env._do_driving() 更新。

用法示例：
1) baseline + 有窗口：python visual.py --mode baseline --baseline-version v1 --render pygame --steps 1000
2) baseline + 服务器导出逐帧：python visual.py --mode baseline --baseline-version v1 --render frames --out frames_v1 --steps 500
3) agent：python visual.py --mode agent --path path/to/model.pt --render pygame --infinite
4) GAIL 策略：python visual.py --mode imitation --path gail_policy.pt --render pygame --fps 1 --infinite
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from environment import JunctionPetriNetEnv
from rewards import base_reward
from utils.petri_net import Parser, get_petri_net

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
GREEN = (50, 200, 50)
YELLOW = (255, 200, 0)
GRAY = (100, 100, 100)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (60, 60, 60)
BLUE = (100, 150, 255)
ORANGE = (255, 165, 0)

# 画布设置（路口区域 + 统计面板）
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900


def _ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def flatten_obs_to_1d(env: JunctionPetriNetEnv, obs: Dict) -> np.ndarray:
    """
    把 env._get_obs() 返回的 dict 观测，按“插入顺序”拼成 1D 向量。

    为什么这么做：
    - `CDQN.act` 里把输入 states 当成 torch.Tensor，并用 states.numpy()[0][:action_space.n] 取前 N 维当“约束掩码”
    - 这要求：观测向量的最前面必须就是“每个变迁是否可用 (0/1)”那一段
    - `environment/petri_net.py::_get_obs()` 正是按这个顺序构建 dict 的（先 transitions mask，再车辆观测）
    """
    # 注意：Python 3.7+ dict 保序；本项目就是依赖这个顺序来“前 N 维 = 变迁可用性”
    parts: List[float] = []
    for k in obs.keys():
        v = obs[k]
        # v 可能是 np.array(scalar) 或 np.array([x]) 或 float
        if isinstance(v, np.ndarray):
            parts.extend([float(x) for x in np.ravel(v)])
        else:
            parts.append(float(v))
    return np.asarray(parts, dtype=np.float32)


class Controller:
    """统一的控制器接口：给定观测，输出一个离散动作（int）"""

    def reset(self) -> None:  # pragma: no cover
        pass

    def act(self, obs: Dict, t: int) -> int:  # pragma: no cover
        raise NotImplementedError


@dataclass
class BaselineController(Controller):
    """
    Baseline 控制器：循环播放固定的动作序列（由变迁名称构成）。

    这里“动作序列”来自您现有的 `baseline.py`（V1 / V2 两个版本）。
    """

    env: JunctionPetriNetEnv
    action_sequence_names: Sequence[str]
    _action_sequence: List[int] = None

    def reset(self) -> None:
        transitions_to_actions = {name: idx for idx, name in enumerate(self.env.actions_to_transitions)}
        self._action_sequence = [transitions_to_actions[name] for name in self.action_sequence_names]

    def act(self, obs: Dict, t: int) -> int:
        if not self._action_sequence:
            # 兜底：如果序列为空，就随机
            return int(np.random.randint(0, self.env.action_space.n))
        return int(self._action_sequence[t % len(self._action_sequence)])


@dataclass
class AgentController(Controller):
    """
    智能体控制器：加载训练好的模型，用贪婪策略选动作。

    关键点：
    - 本项目的 `CDQN.act` 期望输入是 torch.Tensor（扁平化观测）
    - 所以我们把 dict 观测 flatten 后喂给 agent.act
    """

    env: JunctionPetriNetEnv
    model_path: str
    constrained: bool = True

    def __post_init__(self) -> None:
        # 这里延迟导入 torch / skrl，避免用户只跑 baseline 时也要求装齐深度学习依赖
        import torch  # noqa
        from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG  # noqa
        from skrl.memories.torch import RandomMemory  # noqa
        from skrl.envs.torch import wrap_env  # noqa
        from agents.dqn import get_dqn_model  # noqa

        # skrl 的 Agent 通常在 wrap_env 后的环境上构建（内部会处理 device、tensor 转换等）
        wrapped_env = wrap_env(self.env, wrapper="gymnasium")
        memory = RandomMemory(memory_size=1, num_envs=wrapped_env.num_envs)
        cfg = DQN_DEFAULT_CONFIG.copy()
        # 评估模式：不探索
        cfg["exploration"]["timesteps"] = 0
        cfg["exploration"]["final_epsilon"] = 0.0
        cfg["random_timesteps"] = 0

        self._torch = torch
        self._agent = get_dqn_model(env=wrapped_env, memory=memory, cfg=cfg, constrained=self.constrained)
        self._agent.load(self.model_path)
        self._agent.set_running_mode("eval")

    def act(self, obs: Dict, t: int) -> int:
        states_1d = flatten_obs_to_1d(self.env, obs)  # shape: (obs_dim,)
        states = self._torch.from_numpy(states_1d).unsqueeze(0)  # shape: (1, obs_dim)
        with self._torch.no_grad():
            action, _, _ = self._agent.act(states, timestep=t, timesteps=1)
        return int(action.item())


@dataclass
class ImitationController(Controller):
    """
    模仿学习控制器：使用 gail.py 训练出的简单 MLP 策略。

    - 输入：flatten 后的观测向量
    - 输出：每个动作的 logits，取 argmax 作为离散动作
    """

    env: JunctionPetriNetEnv
    model_path: str

    def __post_init__(self) -> None:
        import torch  # 延迟导入

        ckpt = torch.load(self.model_path, map_location="cpu")
        self._torch = torch

        class PolicyMLP(torch.nn.Module):
            def __init__(self, obs_dim: int, act_dim: int):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(obs_dim, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, act_dim),
                )

            def forward(self, x):
                return self.net(x)

        obs_dim = int(ckpt["obs_dim"])
        act_dim = int(ckpt["act_dim"])
        self._policy = PolicyMLP(obs_dim, act_dim)
        self._policy.load_state_dict(ckpt["state_dict"])
        self._policy.eval()

    def act(self, obs: Dict, t: int) -> int:
        x = flatten_obs_to_1d(self.env, obs)
        x_t = self._torch.from_numpy(x).unsqueeze(0)
        with self._torch.no_grad():
            logits = self._policy(x_t)
            action = int(self._torch.argmax(logits, dim=-1).item())
        return action


class PygameRenderer:
    """
    pygame 渲染器（可窗口显示，也可离屏保存帧）。

    - render_mode='pygame' : 开窗口实时看
    - render_mode='frames' : 不开窗口（SDL dummy），每步保存一张 PNG
    """

    def __init__(self, env: JunctionPetriNetEnv, fps: int = 4, render_mode: str = "pygame", out_dir: str = ""):
        self.env = env
        self.fps = fps
        self.render_mode = render_mode
        self.out_dir = out_dir

        # 只有在 frames 模式下才需要输出目录
        if self.render_mode == "frames":
            _ensure_dir(self.out_dir)

        # 延迟导入 pygame，避免环境没装 pygame 时 baseline/text 也被拖死
        import pygame  # noqa

        self.pygame = pygame
        pygame.init()

        if self.render_mode == "frames":
            # 服务器无桌面时：建议用户在命令行设置 SDL_VIDEODRIVER=dummy
            # 这里仍然创建一个 Surface 用于渲染
            self.screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        else:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("交通路口可视化 - Baseline / Agent 控制")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 20)

        # 路口中心位置（左侧 800x800 区域）
        self.center_x = 400
        self.center_y = 400
        self.road_width = 60  # 单条车道宽度

    def close(self) -> None:
        if self.render_mode != "frames":
            self.pygame.quit()

    def _active_places(self) -> List[str]:
        active_places = []
        for place in self.env.net.place():
            if len(place.tokens.items()) > 0:
                active_places.append(place.name)
        return active_places

    def _lights(self) -> Dict[str, str]:
        """
        车道->灯色（简化为红/绿）：
        - 每条车道 LanePetriNetTuple 都绑定一个 place（例如 GreenSN / GreenWE ...）
        - place 上有 token => 对应车道为绿灯
        """
        active_places = set(self._active_places())
        lights = {lane.name: "red" for lane in self.env.lanes}
        for lane in self.env.lanes:
            if lane.place in active_places:
                lights[lane.name] = "green"
        return lights

    def _draw_road(self) -> None:
        pygame = self.pygame
        self.screen.fill((100, 180, 100))  # 草地背景

        intersection_size = self.road_width * 4
        pygame.draw.rect(
            self.screen,
            DARK_GRAY,
            (
                self.center_x - intersection_size // 2,
                self.center_y - intersection_size // 2,
                intersection_size,
                intersection_size,
            ),
        )

        # 南北、东西主路
        pygame.draw.rect(
            self.screen, DARK_GRAY, (self.center_x - self.road_width * 2, 0, self.road_width * 4, WINDOW_HEIGHT)
        )
        pygame.draw.rect(
            self.screen, DARK_GRAY, (0, self.center_y - self.road_width * 2, WINDOW_WIDTH, self.road_width * 4)
        )

        # 车道虚线（黄）
        dash_length = 20
        gap_length = 15
        for lane_offset in [-self.road_width, 0, self.road_width]:
            x = self.center_x + lane_offset
            y = 0
            while y < WINDOW_HEIGHT:
                if abs(y - self.center_y) > intersection_size // 2:
                    pygame.draw.line(self.screen, YELLOW, (x, y), (x, y + dash_length), 2)
                y += dash_length + gap_length

        for lane_offset in [-self.road_width, 0, self.road_width]:
            y = self.center_y + lane_offset
            x = 0
            while x < WINDOW_WIDTH:
                if abs(x - self.center_x) > intersection_size // 2:
                    pygame.draw.line(self.screen, YELLOW, (x, y), (x + dash_length, y), 2)
                x += dash_length + gap_length

    def _draw_traffic_light(self, x: int, y: int, color: str, label: str) -> None:
        pygame = self.pygame
        light_width = 30
        light_height = 80
        pygame.draw.rect(
            self.screen,
            BLACK,
            (x - light_width // 2, y - light_height // 2, light_width, light_height),
            border_radius=5,
        )

        light_radius = 10
        red_color = RED if color == "red" else GRAY
        yellow_color = YELLOW if color == "yellow" else GRAY
        green_color = GREEN if color == "green" else GRAY

        pygame.draw.circle(self.screen, red_color, (x, y - light_height // 2 + 20), light_radius)
        pygame.draw.circle(self.screen, yellow_color, (x, y), light_radius)
        pygame.draw.circle(self.screen, green_color, (x, y + light_height // 2 - 20), light_radius)

        text = self.small_font.render(label, True, WHITE)
        self.screen.blit(text, text.get_rect(center=(x, y + light_height // 2 + 15)))

    def _draw_vehicle(self, x: int, y: int, waiting_time: int) -> None:
        pygame = self.pygame
        car_width = 40
        car_height = 25

        # 等待时间越久，颜色越“危险”
        if waiting_time < 10:
            color = BLUE
        elif waiting_time < 30:
            color = ORANGE
        else:
            color = RED

        pygame.draw.rect(
            self.screen,
            color,
            (x - car_width // 2, y - car_height // 2, car_width, car_height),
            border_radius=3,
        )
        pygame.draw.rect(
            self.screen,
            BLACK,
            (x - car_width // 2, y - car_height // 2, car_width, car_height),
            2,
            border_radius=3,
        )

        if waiting_time > 0:
            t = self.small_font.render(str(waiting_time), True, WHITE)
            self.screen.blit(t, t.get_rect(center=(x, y)))

    def _draw_vehicles(self) -> None:
        """
        只画每条车道前 5 辆车（否则太密），并用“+N”提示剩余。
        车辆对象在 env.lanes[i].vehicles 里，等待时间是 vehicle.time_steps。
        """
        spacing = 50
        for lane in self.env.lanes:
            num = len(lane.vehicles)
            vehicles = lane.vehicles[:5]

            if "north" in lane.name:
                x = self.center_x - (self.road_width * 1.5 if "front" in lane.name else self.road_width * 0.5)
                for i, v in enumerate(vehicles):
                    y = self.center_y - self.road_width * 2 - 50 - i * spacing
                    self._draw_vehicle(int(x), int(y), int(v.time_steps))
                if num > 5:
                    t = self.small_font.render(f"+{num - 5}", True, WHITE)
                    self.screen.blit(t, (x - 15, self.center_y - self.road_width * 2 - 300))

            elif "south" in lane.name:
                x = self.center_x + (self.road_width * 1.5 if "front" in lane.name else self.road_width * 0.5)
                for i, v in enumerate(vehicles):
                    y = self.center_y + self.road_width * 2 + 50 + i * spacing
                    self._draw_vehicle(int(x), int(y), int(v.time_steps))
                if num > 5:
                    t = self.small_font.render(f"+{num - 5}", True, WHITE)
                    self.screen.blit(t, (x - 15, self.center_y + self.road_width * 2 + 300))

            elif "west" in lane.name:
                y = self.center_y + (self.road_width * 1.5 if "front" in lane.name else self.road_width * 0.5)
                for i, v in enumerate(vehicles):
                    x = self.center_x - self.road_width * 2 - 50 - i * spacing
                    self._draw_vehicle(int(x), int(y), int(v.time_steps))
                if num > 5:
                    t = self.small_font.render(f"+{num - 5}", True, WHITE)
                    self.screen.blit(t, (self.center_x - self.road_width * 2 - 300, y - 10))

            elif "east" in lane.name:
                y = self.center_y - (self.road_width * 1.5 if "front" in lane.name else self.road_width * 0.5)
                for i, v in enumerate(vehicles):
                    x = self.center_x + self.road_width * 2 + 50 + i * spacing
                    self._draw_vehicle(int(x), int(y), int(v.time_steps))
                if num > 5:
                    t = self.small_font.render(f"+{num - 5}", True, WHITE)
                    self.screen.blit(t, (self.center_x + self.road_width * 2 + 300, y - 10))

    def _draw_lights(self, lights: Dict[str, str]) -> None:
        offset = self.road_width * 2 + 40
        self._draw_traffic_light(self.center_x - int(self.road_width * 1.5), self.center_y - offset, lights["north_front"], "北直")
        self._draw_traffic_light(self.center_x - int(self.road_width * 0.5), self.center_y - offset, lights["north_left"], "北左")

        self._draw_traffic_light(self.center_x + int(self.road_width * 1.5), self.center_y + offset, lights["south_front"], "南直")
        self._draw_traffic_light(self.center_x + int(self.road_width * 0.5), self.center_y + offset, lights["south_left"], "南左")

        self._draw_traffic_light(self.center_x - offset, self.center_y + int(self.road_width * 1.5), lights["west_front"], "西直")
        self._draw_traffic_light(self.center_x - offset, self.center_y + int(self.road_width * 0.5), lights["west_left"], "西左")

        self._draw_traffic_light(self.center_x + offset, self.center_y - int(self.road_width * 1.5), lights["east_front"], "东直")
        self._draw_traffic_light(self.center_x + offset, self.center_y - int(self.road_width * 0.5), lights["east_left"], "东左")

    def _draw_panel(
        self,
        t: int,
        total_reward: float,
        cars_driven: int,
        constraints_broken: int,
        avg_wait: Optional[float],
        last_action_name: str,
    ) -> None:
        pygame = self.pygame
        panel_x, panel_y, panel_w, panel_h = 850, 50, 500, 800
        pygame.draw.rect(self.screen, WHITE, (panel_x, panel_y, panel_w, panel_h), border_radius=10)
        pygame.draw.rect(self.screen, BLACK, (panel_x, panel_y, panel_w, panel_h), 3, border_radius=10)

        title = self.title_font.render("统计面板", True, BLACK)
        self.screen.blit(title, (panel_x + 20, panel_y + 20))

        stats = [
            f"步数: {t}",
            f"累计奖励: {total_reward:.1f}",
            f"通行车辆: {cars_driven}",
            f"约束违反: {constraints_broken}",
            f"平均等待时间: {avg_wait:.2f}" if avg_wait is not None else "平均等待时间: N/A",
            "",
            f"当前动作: {last_action_name}",
            "",
            "当前激活库所（token 所在 place）:",
        ]

        y = panel_y + 80
        for s in stats:
            self.screen.blit(self.font.render(s, True, BLACK), (panel_x + 25, y))
            y += 32

        for place in self._active_places():
            self.screen.blit(self.small_font.render(f"- {place}", True, GREEN), (panel_x + 35, y))
            y += 24

        y += 10
        self.screen.blit(self.font.render("各车道排队:", True, BLACK), (panel_x + 25, y))
        y += 30
        for lane in self.env.lanes:
            line = f"{lane.name:>12}: {len(lane.vehicles):>3} 辆 | 最长等待 {lane.max_time():>3}"
            self.screen.blit(self.small_font.render(line, True, BLACK), (panel_x + 25, y))
            y += 22

        help_y = panel_y + panel_h - 110
        help_texts = ["车辆颜色:", "蓝 <10步", "橙 10~30步", "红 >30步"]
        for i, ht in enumerate(help_texts):
            self.screen.blit(self.small_font.render(ht, True, DARK_GRAY), (panel_x + 25, help_y + 22 * i))

    def render_step(
        self,
        t: int,
        total_reward: float,
        cars_driven: int,
        constraints_broken: int,
        waiting_times: List[int],
        last_action_name: str,
    ) -> None:
        # 事件处理（只有开窗口时需要）
        if self.render_mode != "frames":
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE:
                    raise KeyboardInterrupt

        self._draw_road()
        lights = self._lights()
        self._draw_vehicles()
        self._draw_lights(lights)

        avg_wait = float(np.mean(waiting_times)) if waiting_times else None
        self._draw_panel(t, total_reward, cars_driven, constraints_broken, avg_wait, last_action_name)

        if self.render_mode == "frames":
            # 保存逐帧图片（4 位编号，便于 ffmpeg -i frame_%04d.png 合成视频）
            filename = os.path.join(self.out_dir, f"frame_{t:04d}.png")
            self.pygame.image.save(self.screen, filename)
        else:
            self.pygame.display.flip()
            self.clock.tick(self.fps)


def run_episode(
    env: JunctionPetriNetEnv,
    controller: Controller,
    renderer: Optional[PygameRenderer],
    max_steps: int,
    infinite: bool = False,
) -> None:
    """
    把“控制器(选动作)” + “环境(step)” + “渲染器(画图)”串起来的主循环。

    这段就是您问的“怎么和 env 配合”的核心：
    - controller.act(obs, t) 产出 action（离散整数）
    - env.step(action) 内部会：
      1) 把 action 映射成 Petri 网变迁名（env.actions_to_transitions[action]）
      2) 检查该变迁是否有 modes（可触发），有则 fire（token 流动），否则 success=False
      3) 根据 token 所在 place（即当前相位）让对应车道通行，并生成/等待车辆
      4) 计算 reward，返回 info（包含 success / num_cars_driven / waiting_times 等）
    """
    obs, _ = env.reset()
    controller.reset()

    terminated = False
    total_reward = 0.0
    cars_driven = 0
    constraints_broken = 0
    all_waiting_times: List[int] = []

    t = 0
    while True:
        if terminated:
            break
        if (not infinite) and t >= max_steps:
            break

        action = controller.act(obs, t)
        obs, reward, terminated, _truncated, info = env.step(action)

        total_reward += float(reward)
        cars_driven += int(info.get("num_cars_driven", 0))
        if not info.get("success", True):
            constraints_broken += 1
        if info.get("waiting_times"):
            all_waiting_times.extend(info["waiting_times"])

        if renderer is not None:
            renderer.render_step(
                t=t,
                total_reward=total_reward,
                cars_driven=cars_driven,
                constraints_broken=constraints_broken,
                waiting_times=all_waiting_times,
                last_action_name=env.actions_to_transitions[action],
            )
        t += 1


def build_env() -> JunctionPetriNetEnv:
    """
    构建环境（和 baseline.py / evaluation.py 对齐）。
    """
    env = JunctionPetriNetEnv(
        reward_function=base_reward,
        net=get_petri_net("data/traffic-scenario.PNPRO", type=Parser.PNPRO),
        transitions_to_obs=True,
        places_to_obs=False,
    )
    env.reset()
    return env


def main():
    """
    CLI 入口：选择控制源（baseline/agent）和渲染方式（pygame/frames/none）。
    """
    parser = argparse.ArgumentParser(prog="交通路口可视化工具（baseline vs agent）")
    parser.add_argument(
        "--mode",
        choices=["baseline", "agent", "imitation"],
        default="baseline",
        help="选择控制源：baseline / agent(DQN) / imitation(gail.py 训练的MLP)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="模型路径（--mode agent 或 imitation 时必填）",
    )
    parser.add_argument("--baseline-version", choices=["v1", "v2"], default="v1", help="baseline 使用哪个序列")
    parser.add_argument("--steps", type=int, default=500, help="最大仿真步数")
    parser.add_argument("--fps", type=int, default=4, help="pygame 刷新帧率")
    parser.add_argument(
        "--render",
        choices=["pygame", "frames", "none"],
        default="pygame",
        help="渲染方式：pygame=开窗口；frames=逐帧PNG（适合服务器）；none=不渲染只跑逻辑",
    )
    parser.add_argument("--out", type=str, default="frames", help="frames 模式输出目录")
    parser.add_argument("--infinite", action="store_true", help="忽略 --steps，一直运行直到手动关闭/CTRL+C")

    args = parser.parse_args()

    env = build_env()

    # baseline 动作序列（与 baseline.py 保持一致）
    action_sequence_names_v1 = [
        "RtoGwnes",
        "GtoRwnes",
        "RtoGswne",
        "GtoRswne",
        "RtoGsn",
        "GtoRsn",
        "RtoGwe",
        "GtoRwe",
    ]
    action_sequence_names_v2 = [
        "RtoGwnes",
        "GtoRwnes",
        "RtoGswne",
        "GtoRswne",
        "RtoGsn",
        "None",
        "GtoRsn",
        "RtoGwe",
        "None",
        "GtoRwe",
    ]

    if args.mode == "baseline":
        names = action_sequence_names_v1 if args.baseline_version == "v1" else action_sequence_names_v2
        controller: Controller = BaselineController(env=env, action_sequence_names=names)
        print(f"[visual] 模式: baseline ({args.baseline_version})")
        print(f"[visual] 动作序列(变迁名): {list(names)}")
    elif args.mode == "agent":
        if not args.path:
            print("错误：--mode agent 时必须提供 --path <模型文件.pt>")
            sys.exit(2)
        controller = AgentController(env=env, model_path=args.path, constrained=True)
        print(f"[visual] 模式: agent (model={args.path})")
    else:  # imitation
        if not args.path:
            print("错误：--mode imitation 时必须提供 --path <模仿学习模型.pt>")
            sys.exit(2)
        controller = ImitationController(env=env, model_path=args.path)
        print(f"[visual] 模式: imitation (model={args.path})")

    renderer: Optional[PygameRenderer] = None
    if args.render in ("pygame", "frames"):
        if args.render == "frames":
            # 提示：无桌面服务器需要 dummy driver
            # Linux 示例：export SDL_VIDEODRIVER=dummy
            _ensure_dir(args.out)
            print(f"[visual] 渲染: frames (输出目录: {args.out})")
        else:
            print("[visual] 渲染: pygame (按 ESC 或关闭窗口退出)")

        renderer = PygameRenderer(env=env, fps=args.fps, render_mode=args.render, out_dir=args.out)
    else:
        print("[visual] 渲染: none")

    try:
        run_episode(
            env=env,
            controller=controller,
            renderer=renderer,
            max_steps=int(args.steps),
            infinite=bool(args.infinite),
        )
    except KeyboardInterrupt:
        print("[visual] 用户中断退出")
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
