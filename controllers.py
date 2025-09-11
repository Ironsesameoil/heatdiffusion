"""
controllers.py

로봇을 제어하기 위한 다양한 제어기 클래스를 정의합니다.
모든 컨트롤러는 BaseController를 상속받아 일관된 인터페이스를 제공합니다.
"""

import torch
import math
from abc import ABC, abstractmethod
from typing import List

# Isaac Lab 컨트롤러 (비교/참조용)
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
# Isaac Sim 모션 생성 라이브러리
#from isaacsim.robot_motion.motion_generation import DifferentialController


class BaseController(ABC):
    """
    모든 컨트롤러 클래스가 상속받을 추상 기반 클래스입니다.
    """
    @abstractmethod
    def initialize(self):
        """컨트롤러에 필요한 초기화 작업을 수행합니다."""
        pass

    @abstractmethod
    def compute_action(self, robot_position: torch.Tensor, robot_yaw: float, target_position: torch.Tensor) -> torch.Tensor:
        """
        로봇의 현재 상태와 목표 지점을 바탕으로 제어 명령(action)을 계산합니다.
        """
        pass

# ===============================================================
# -------------------- 컨트롤러 구현부 ---------------------------
# ===============================================================

### 1. AnalyticController (P-Controller) ###

class AnalyticController(BaseController):
    """
    미분 구동 로봇의 역기구학을 직접 계산하는 수동 비례(P) 제어기입니다.
    """
    def __init__(self, num_dof: int, wheel_radius: float, track_width: float, 
                 left_wheel_indices: List[int], right_wheel_indices: List[int], 
                 gains: dict, device: str):
        self.num_dof = num_dof
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        self.left_indices = left_wheel_indices
        self.right_indices = right_wheel_indices
        self.gains = gains
        self.device = device
        print("🕹️  AnalyticController (P-Control)が初期化されました.")

    def initialize(self):
        pass

    def compute_action(self, robot_position: torch.Tensor, robot_yaw: float, target_position: torch.Tensor, dt: float) -> torch.Tensor:
        dx = target_position[0] - robot_position[0]
        dy = target_position[1] - robot_position[1]
        target_yaw = torch.atan2(dy, dx)
        yaw_err = (target_yaw - robot_yaw + math.pi) % (2 * math.pi) - math.pi
        dist_to_target = torch.sqrt(dx**2 + dy**2)
        v_cmd = torch.clamp(self.gains['K_V'] * dist_to_target, -self.gains['V_MAX'], self.gains['V_MAX'])
        w_cmd = torch.clamp(self.gains['K_W'] * yaw_err, -self.gains['W_MAX'], self.gains['W_MAX'])
        w_left = (v_cmd - 0.5 * w_cmd * self.track_width) / self.wheel_radius
        w_right = (v_cmd + 0.5 * w_cmd * self.track_width) / self.wheel_radius
        action = torch.zeros(self.num_dof, device=self.device)
        for idx in self.left_indices: action[idx] = w_left
        for idx in self.right_indices: action[idx] = w_right
        return action

### 2. PIController ###

class PIController(BaseController):
    """
    정상상태 오차를 줄이기 위해 적분(I) 항이 추가된 비례-적분(PI) 제어기입니다.
    """
    def __init__(self, num_dof: int, wheel_radius: float, track_width: float,
                 left_wheel_indices: List[int], right_wheel_indices: List[int],
                 gains: dict, device: str):
        self.num_dof = num_dof
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        self.left_indices = left_wheel_indices
        self.right_indices = right_wheel_indices
        self.gains = gains
        self.device = device
        self.integral_dist_err = 0.0
        self.integral_yaw_err = 0.0
        print("🕹️  PIController (P+I Control)が初期化されました.")

    def initialize(self):
        self.integral_dist_err = 0.0
        self.integral_yaw_err = 0.0

    def compute_action(self, robot_position: torch.Tensor, robot_yaw: float, target_position: torch.Tensor, dt: float) -> torch.Tensor:
        dx = target_position[0] - robot_position[0]
        dy = target_position[1] - robot_position[1]
        target_yaw = torch.atan2(dy, dx)
        yaw_err = (target_yaw - robot_yaw + math.pi) % (2 * math.pi) - math.pi
        dist_to_target = torch.sqrt(dx**2 + dy**2)
        self.integral_dist_err += dist_to_target * dt
        self.integral_yaw_err += yaw_err * dt
        max_integral_v = self.gains.get("I_V_MAX", 1.0)
        max_integral_w = self.gains.get("I_W_MAX", 1.0)
        self.integral_dist_err = torch.clamp(self.integral_dist_err, -max_integral_v, max_integral_v)
        self.integral_yaw_err = torch.clamp(self.integral_yaw_err, -max_integral_w, max_integral_w)
        p_v = self.gains['K_V'] * dist_to_target
        p_w = self.gains['K_W'] * yaw_err
        i_v = self.gains['K_I_V'] * self.integral_dist_err
        i_w = self.gains['K_I_W'] * self.integral_yaw_err
        v_cmd = torch.clamp(p_v + i_v, -self.gains['V_MAX'], self.gains['V_MAX'])
        w_cmd = torch.clamp(p_w + i_w, -self.gains['W_MAX'], self.gains['W_MAX'])
        w_left = (v_cmd - 0.5 * w_cmd * self.track_width) / self.wheel_radius
        w_right = (v_cmd + 0.5 * w_cmd * self.track_width) / self.wheel_radius
        action = torch.zeros(self.num_dof, device=self.device)
        for idx in self.left_indices: action[idx] = w_left
        for idx in self.right_indices: action[idx] = w_right
        return action

### 3. PIDController (NEW) ###

class PIDController(BaseController):
    """
    오버슈팅을 줄이고 안정성을 높이기 위해 미분(D) 항이 추가된 비례-적분-미분(PID) 제어기입니다.
    """
    def __init__(self, num_dof: int, wheel_radius: float, track_width: float,
                 left_wheel_indices: List[int], right_wheel_indices: List[int],
                 gains: dict, device: str):
        self.num_dof = num_dof
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        self.left_indices = left_wheel_indices
        self.right_indices = right_wheel_indices
        self.gains = gains
        self.device = device
        
        # Integral and Derivative terms
        self.integral_dist_err = 0.0
        self.integral_yaw_err = 0.0
        self.prev_dist_err = 0.0
        self.prev_yaw_err = 0.0
        print("🕹️  PIDController ")

    def initialize(self):
        # Reset all stateful terms
        self.integral_dist_err = 0.0
        self.integral_yaw_err = 0.0
        self.prev_dist_err = 0.0
        self.prev_yaw_err = 0.0

    def compute_action(self, robot_position: torch.Tensor, robot_yaw: float, target_position: torch.Tensor, dt: float) -> torch.Tensor:
        if dt <= 0: return torch.zeros(self.num_dof, device=self.device)

        # 1. Calculate current errors
        dx = target_position[0] - robot_position[0]
        dy = target_position[1] - robot_position[1]
        target_yaw = torch.atan2(dy, dx)
        yaw_err = (target_yaw - robot_yaw + math.pi) % (2 * math.pi) - math.pi
        dist_to_target = torch.sqrt(dx**2 + dy**2)

        # 2. Update Integral terms (with anti-windup)
        self.integral_dist_err += dist_to_target * dt
        self.integral_yaw_err += yaw_err * dt
        max_integral_v = self.gains.get("I_V_MAX", 1.0)
        max_integral_w = self.gains.get("I_W_MAX", 1.0)
        self.integral_dist_err = torch.clamp(self.integral_dist_err, -max_integral_v, max_integral_v)
        self.integral_yaw_err = torch.clamp(self.integral_yaw_err, -max_integral_w, max_integral_w)

        # 3. Calculate Derivative terms
        derivative_dist = (dist_to_target - self.prev_dist_err) / dt
        derivative_yaw = (yaw_err - self.prev_yaw_err) / dt

        # 4. Update previous errors for next cycle
        self.prev_dist_err = dist_to_target
        self.prev_yaw_err = yaw_err

        # 5. Calculate PID command velocities
        p_v = self.gains['K_V'] * dist_to_target
        p_w = self.gains['K_W'] * yaw_err
        i_v = self.gains['K_I_V'] * self.integral_dist_err
        i_w = self.gains['K_I_W'] * self.integral_yaw_err
        d_v = self.gains['K_D_V'] * derivative_dist
        d_w = self.gains['K_D_W'] * derivative_yaw
        
        v_cmd = torch.clamp(p_v + i_v + d_v, -self.gains['V_MAX'], self.gains['V_MAX'])
        w_cmd = torch.clamp(p_w + i_w + d_w, -self.gains['W_MAX'], self.gains['W_MAX'])

        # 6. Inverse kinematics
        w_left = (v_cmd - 0.5 * w_cmd * self.track_width) / self.wheel_radius
        w_right = (v_cmd + 0.5 * w_cmd * self.track_width) / self.wheel_radius
        action = torch.zeros(self.num_dof, device=self.device)
        for idx in self.left_indices: action[idx] = w_left
        for idx in self.right_indices: action[idx] = w_right
        return action


'''
### 2. MotionGenDifferentialController (Isaac Sim 기본 라이브러리) ###

class MotionGenDifferentialController(BaseController):
    """
    omni.isaac.motion_generation.DifferentialController를 래핑(wrapping)한 클래스입니다.
    """
    def __init__(self, num_dof: int, wheel_radius: float, track_width: float, 
                 left_wheel_indices: List[int], right_wheel_indices: List[int], 
                 gains: dict, device: str):
        self.num_dof = num_dof
        self.left_indices = left_wheel_indices
        self.right_indices = right_wheel_indices
        self.gains = gains
        self.device = device

        # Isaac Sim 모션 생성 라이브러리의 컨트롤러 인스턴스 생성
        self.motion_gen_controller = DifferentialController(
            name="default_differential_controller",
            wheel_radius=wheel_radius,
            wheel_base=track_width
        )
        print("🕹️ MotionGenDifferentialController (Isaac Sim 라이브러리)가 초기화되었습니다.")

    def initialize(self):
        pass

    def compute_action(self, robot_position: torch.Tensor, robot_yaw: float, target_position: torch.Tensor) -> torch.Tensor:
        # 1. 목표 선속도(v)와 각속도(w) 계산
        dx = target_position[0] - robot_position[0]
        dy = target_position[1] - robot_position[1]
        target_yaw = torch.atan2(dy, dx)
        yaw_err = (target_yaw - robot_yaw + math.pi) % (2 * math.pi) - math.pi
        dist_to_target = torch.sqrt(dx**2 + dy**2)

        v_cmd = torch.clamp(self.gains['K_V'] * dist_to_target, -self.gains['V_MAX'], self.gains['V_MAX'])
        w_cmd = torch.clamp(self.gains['K_W'] * yaw_err, -self.gains['W_MAX'], self.gains['W_MAX'])

        # 2. 컨트롤러의 forward 함수를 호출하여 좌/우 바퀴 목표 각속도 계산
        # 참고: 이 컨트롤러의 출력은 [w_left, w_right]가 아닌 [v_left, v_right] (바퀴의 선속도) 입니다.
        # 따라서 바퀴 반지름으로 나누어 각속도로 변환해야 합니다.
        wheel_velocities = self.motion_gen_controller.forward(command=[v_cmd, w_cmd])
        w_left = wheel_velocities[0] / self.gains['wheel_radius']
        w_right = wheel_velocities[1] / self.gains['wheel_radius']

        # 3. 전체 조인트에 대한 속도 텐서 생성 후, 해당 인덱스에 값 할당
        action = torch.zeros(self.num_dof, device=self.device)
        for idx in self.left_indices:
            action[idx] = w_left
        for idx in self.right_indices:
            action[idx] = w_right
            
        return action
'''

### 3. IsaacLabIKController (Isaac Lab 권장 방식, 참조용) ###

class IsaacLabIKController(BaseController):
    """
    omni.isaac.lab.controllers.DifferentialIKController를 래핑한 클래스입니다.
    Isaac Lab 환경에서는 이 컨트롤러가 가장 효율적이고 권장됩니다.
    """
    def __init__(self, prim_path: str, gains: dict, device: str):
        self.gains = gains
        self.device = device

        # [THE CORRECT METHOD]
        # Step 1: Create an empty Cfg object first.
        ik_cfg = DifferentialIKControllerCfg()
        
        # Step 2: Assign values to its properties directly, line by line.
        ik_cfg.command_type = "velocity"
        ik_cfg.joint_names_expr = [".*wheel_joint"]
        ik_cfg.body_name = "chassis"
        
        # Create the controller instance.
        self.ik_controller = DifferentialIKController(cfg=ik_cfg, prim_paths_expr=prim_path)
        print("🕹️ IsaacLabIKController (Correct Method) has been initialized.")

    def initialize(self):
        self.ik_controller.initialize()

    def compute_action(self, robot_position: torch.Tensor, robot_yaw: float, target_position: torch.Tensor) -> torch.Tensor:
        # 1. Calculate the target linear (v) and angular (w) velocities
        dx = target_position[0] - robot_position[0]
        dy = target_position[1] - robot_position[1]
        target_yaw = torch.atan2(dy, dx)
        yaw_err = (target_yaw - robot_yaw + math.pi) % (2 * math.pi) - math.pi
        dist_to_target = torch.sqrt(dx**2 + dy**2)

        v_cmd = torch.clamp(self.gains['K_V'] * dist_to_target, -self.gains['V_MAX'], self.gains['V_MAX'])
        w_cmd = torch.clamp(self.gains['K_W'] * yaw_err, -self.gains['W_MAX'], self.gains['W_MAX'])

        # 2. Pass the calculated v and w to the Isaac Lab controller to get the full action tensor
        cmd_vel = torch.tensor([[v_cmd, 0, 0, 0, 0, w_cmd]], device=self.device)
        joint_velocities = self.ik_controller.forward(cmd_vel)
        return joint_velocities.squeeze()