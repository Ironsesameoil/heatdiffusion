"""
waypoint.py

이 파일은 수동으로 정의된 웨이포인트 경로 데이터를 저장하고,
경로 진행 상황을 추적하는 WaypointManager 클래스를 정의합니다.
"""

import torch
from typing import List, Tuple

# ===============================================================
# ------------------ ⚙️ 웨이포인트 경로 정의 -------------------
# ===============================================================
# ✅✅✅ 여기에 로봇들의 경로를 직접 작성, 추가, 삭제하세요. ✅✅✅
# 시뮬레이션은 여기에 정의된 경로의 수만큼 로봇을 자동으로 생성합니다.

WAYPOINTS_TENSOR = torch.tensor([
    # Robot 0의 경로
    [ 
        [2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [0.0, 0.0]
    ],
    # Robot 1의 경로
    [ 
        [3.0, -2.0], [7.0, -2.0], [7.0, 2.0], [3.0, 2.0], [5.0, 0.0]
    ],
    # Robot 2의 경로
    [
        [0.0, 8.0], [3.0, 5.0], [0.0, 2.0], [-3.0, 5.0], [0.0, 5.0]
    ]
    # 새로운 로봇을 추가하려면 여기에 새 경로 리스트를 추가하세요.
    # 예: [ [x1, y1], [x2, y2], ... ],

], dtype=torch.float32)


# In waypoint.py, replace the entire WaypointManager class

class WaypointManager:
    """
    Look-ahead (Pure Pursuit) 방식의 경로 추종 로직을 관리합니다.
    중간 지점은 부드럽게 통과하고, 최종 목적지에는 정확히 정지합니다.
    """
    def __init__(self, waypoints: List[Tuple[float, float]], device: str, 
                 goal_tolerance: float = 0.1, robot_index: int = 0, look_ahead_distance: float = 0.5):
        if len(waypoints) < 2:
            raise ValueError("경로는 최소 2개의 웨이포인트가 필요합니다.")

        self.waypoints_cpu = waypoints
        self.waypoints_gpu = torch.tensor(waypoints, dtype=torch.float32, device=device)
        self.device = device
        self.goal_tolerance = goal_tolerance
        self.robot_index = robot_index
        self.look_ahead_distance = look_ahead_distance
        
        self.num_waypoints = len(waypoints)
        self.current_segment_index = 0
        self._mission_complete = False
        print(f"🎯 [Robot {self.robot_index}] Advanced WaypointManager가 {self.num_waypoints}개 지점으로 초기화되었습니다 (Look-ahead: {self.look_ahead_distance}m).")

    def update_state(self, robot_position: torch.Tensor):
        """로봇의 위치에 따라 경로 추종 상태(현재 세그먼트, 임무 완료 여부)를 업데이트합니다."""
        if self._mission_complete:
            return

        # 1. 최종 목표에 도달했는지 먼저 확인
        final_waypoint = self.waypoints_gpu[-1]
        dist_to_final = torch.norm(robot_position[:2] - final_waypoint)
        if dist_to_final < self.goal_tolerance:
            self._mission_complete = True
            print(f"🎉 [NAV-{self.robot_index}] 최종 목적지 도달! 임무 완료!")
            return

        # 2. 다음 경로 세그먼트로 전환할지 확인 (마지막 세그먼트가 아닐 때만)
        is_on_last_segment = self.current_segment_index >= self.num_waypoints - 2
        if not is_on_last_segment:
            next_waypoint = self.waypoints_gpu[self.current_segment_index + 1]
            dist_to_next = torch.norm(robot_position[:2] - next_waypoint)
            
            # 다음 웨이포인트에 충분히 가까워지면 세그먼트 인덱스를 업데이트
            if dist_to_next < self.look_ahead_distance:
                self.current_segment_index += 1
                next_wp_cpu = self.waypoints_cpu[self.current_segment_index + 1]
                print(f"[NAV-{self.robot_index}] 경로 세그먼트 업데이트! 다음 웨이포인트 -> {next_wp_cpu}")

    def get_target(self, robot_position: torch.Tensor) -> torch.Tensor:
        """
        현재 로봇의 상태에 따라 적절한 목표 지점을 반환합니다.
        """
        # 마지막 경로 세그먼트를 추종 중이라면, 최종 목적지를 목표로 설정
        is_on_last_segment = self.current_segment_index >= self.num_waypoints - 2
        if is_on_last_segment:
            return self.waypoints_gpu[-1]

        # 중간 경로 세그먼트에서는 Look-ahead 지점을 목표로 설정
        p1 = self.waypoints_gpu[self.current_segment_index]
        p2 = self.waypoints_gpu[self.current_segment_index + 1]
        
        line_vec = p2 - p1
        line_norm = torch.norm(line_vec)
        if line_norm < 1e-6:
            return p2

        p1_to_robot_vec = robot_position - p1
        t = torch.dot(p1_to_robot_vec, line_vec) / (line_norm**2)
        t = torch.clamp(t, 0.0, 1.0)
        
        closest_point = p1 + t * line_vec
        
        look_ahead_target = closest_point + self.look_ahead_distance * (line_vec / line_norm)
        return look_ahead_target

    def is_mission_complete(self) -> bool:
        """이 로봇의 임무가 완전히 종료되었는지 여부를 반환합니다."""
        return self._mission_complete

    def reset(self):
        self.current_segment_index = 0
        self._mission_complete = False
        print(f"🔄 [Robot {self.robot_index}] Waypoint 추적을 리셋했습니다.")


'''
class WaypointManager:
    """
    단일 웨이포인트 경로를 받아, 현재 목표 지점을 추적하고
    목표 도달 여부를 판단하는 로직을 관리합니다.
    """
    def __init__(self, waypoints: List[Tuple[float, float]], device: str, goal_tolerance: float = 0.08, robot_index: int = 0):
        if not waypoints:
            raise ValueError("웨이포인트 목록은 비어 있을 수 없습니다.")

        self.waypoints_cpu = waypoints
        self.waypoints_gpu = torch.tensor(waypoints, dtype=torch.float32, device=device)
        self.device = device
        self.goal_tolerance = goal_tolerance
        self.robot_index = robot_index # 디버깅 메시지에 사용할 인덱스
        
        self.num_waypoints = len(waypoints)
        self.current_index = 0
        print(f"🎯 [Robot {self.robot_index}] WaypointManager가 {self.num_waypoints}개 지점으로 구성된 경로로 초기화되었습니다.")

    def update_waypoint_if_reached(self, robot_position: torch.Tensor) -> bool:
        if self.is_finished():
            return False
        
        target_pos = self.get_current_target()
        target_pos_on_device = target_pos.to(robot_position.device)
        distance = torch.norm(robot_position[:2] - target_pos_on_device)
        
        if distance < self.goal_tolerance:
            self.current_index += 1
            if not self.is_finished():
                next_target = self.waypoints_cpu[self.current_index]
                print(f"[NAV-{self.robot_index}] 목표 도달! 다음 -> {next_target}")
            else:
                print(f"[NAV-{self.robot_index}] 최종 웨이포인트 도달! 임무 완료!")
            return True
        return False

    def get_current_target(self) -> torch.Tensor:
        if self.is_finished():
            return self.waypoints_gpu[-1]
        return self.waypoints_gpu[self.current_index]

    def is_finished(self) -> bool:
        return self.current_index >= self.num_waypoints

    def reset(self):
        self.current_index = 0
        print(f"🔄 [Robot {self.robot_index}] Waypoint 추적을 리셋했습니다.")
'''