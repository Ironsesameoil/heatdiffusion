"""
robot.py

이 파일은 시뮬레이션에 사용될 로봇을 클래스로 정의합니다.
"""

import torch
import math

from omni.isaac.lab.assets import ArticulationCfg, Articulation
from omni.isaac.lab.sim.spawners.from_files import UsdFileCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg

class Dingo:
    USD_PATH = "omniverse://localhost/NVIDIA/Assets/Isaac/5.0/Isaac/Robots/Clearpath/Dingo/dingo.usd"
    WHEEL_RADIUS = 0.1
    TRACK_WIDTH = 0.45

    def __init__(self, prim_path: str, robot_index: int, all_waypoints: torch.Tensor, name: str = "Dingo", position: tuple = (0.0, 0.0, 0.15)):
        self.prim_path = prim_path
        self.name = f"{name}_{robot_index}"
        
        if robot_index >= all_waypoints.shape[0]:
            raise IndexError(f"robot_index({robot_index})가 사용 가능한 경로 수({all_waypoints.shape[0]})를 벗어났습니다.")
        
        self.path_gpu = all_waypoints[robot_index]
        self.path_cpu = self.path_gpu.tolist()
        
        self.cfg = self._create_articulation_cfg(position)
        self.articulation = Articulation(cfg=self.cfg)

    def _create_articulation_cfg(self, position: tuple) -> ArticulationCfg:
        return ArticulationCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.USD_PATH,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=position,
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
            actuators={
                "wheels": ImplicitActuatorCfg(
                    joint_names_expr=[".*wheel.*"],
                    stiffness=0.0,
                    damping=120.0,
                    effort_limit=600.0,
                )
            },
        )

    def get_position(self) -> torch.Tensor:
        """
        로봇의 현재 월드 좌표계 위치 [x, y, z]를 버퍼에서 직접 읽어 반환합니다.
        """
        # self._position을 반환하는 대신 버퍼에서 직접 최신 값을 읽습니다.
        return self.articulation.data.root_pos_w[0]

    def get_yaw(self) -> float:
        """
        로봇의 현재 Z축 회전값(yaw)을 버퍼에서 직접 계산하여 반환합니다.
        """
        root_quat_w = self.articulation.data.root_quat_w[0]
        qw, qx, qy, qz = root_quat_w
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return yaw.item()

    def apply_action(self, action: torch.Tensor):
        self.articulation.set_joint_velocity_target(action)
        self.articulation.write_data_to_sim()

    def stop(self):
        num_joints = len(self.articulation.joint_names)
        zero_velocities = torch.zeros(num_joints, device=self.articulation.device)
        self.apply_action(zero_velocities)

    def get_path(self) -> list:
        return self.path_cpu