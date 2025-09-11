# main.py
"""
모든 모듈을 조립하고 시뮬레이션을 실행하는 메인 파일입니다.
"""
import torch

# Isaac Sim 앱 런처 및 설정
from omni.isaac.lab.app import AppLauncher

# 간단한 AppLauncher 초기화
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.debug_draw import _debug_draw

# 직접 만든 모듈들 임포트
from .sim_setup import SimulationSetup
from .robot import Dingo
from .waypoint import WaypointManager, WAYPOINTS_TENSOR
from .controllers import AnalyticController, PIController, PIDController , IsaacLabIKController  # ← IK 컨트롤러 임포트 추가

# ===============================================================
# ---------------------- ⚙️ 전역 설정 ----------------------------
# ===============================================================

# 사용 가능 모드: "ANALYTIC" | "PI" | "PID" | "IK"
CONTROL_MODE = "PID" # <-- 이제 "IK" 모드를 사용할 수 있습니다.

GOAL_TOLERANCE = 0.1
GAINS = {
    "K_V": 3.0,     # Proportional gain
    "K_W": 4.0,     # Proportional gain
    "K_I_V": 0.05,  # Integral gain
    "K_I_W": 0.2,   # Integral gain
    "K_D_V": 0.1,   # Derivative gain (dampens linear motion)
    "K_D_W": 0.3,   # Derivative gain (dampens turning)
    "V_MAX": 4.0,
    "W_MAX": 7.0,
}

# 로봇이 경로를 따라갈 때 얼마나 앞을 내다볼지 결정 (미터 단위)
# 이 값이 클수록 코너를 더 부드럽게 돌지만, 경로를 벗어날 수 있습니다.
LOOK_AHEAD_DISTANCE = 0.2

# ===============================================================
# -------------------- ▶️ 메인 실행 로직 ---------------------------
# ===============================================================


if __name__ == "__main__":
    # 시뮬레이션 설정
    sim_cfg = SimulationCfg(dt=1.0 / 120.0, device="cuda:0")
    sim_setup = SimulationSetup(sim_cfg=sim_cfg)
    sim_setup.setup_scene()

    # --- Step 1: 로봇 객체와 웨이포인트 매니저 생성 ---
    # waypoint.py에서 생성된 경로 수에 따라 로봇 수를 자동으로 결정
    NUM_ROBOTS = WAYPOINTS_TENSOR.shape[0]
    print(f"🤖 총 {NUM_ROBOTS}대의 로봇을 생성합니다...")
    
    robots = []
    waypoint_managers = []

    for i in range(NUM_ROBOTS):
        robot_prim_path = f"/World/Dingo_{i}"
        
        # 각 로봇의 시작 위치를 해당 경로의 첫 번째 웨이포인트로 설정
        start_waypoint = WAYPOINTS_TENSOR[i][0]
        start_position = (float(start_waypoint[0]), float(start_waypoint[1]), 0.15)
        
        robot = Dingo(
            prim_path=robot_prim_path,
            robot_index=i,
            all_waypoints=WAYPOINTS_TENSOR,
            position=start_position,
        )
        robots.append(robot)

        waypoint_manager = WaypointManager(
            waypoints=robot.get_path(),
            device=sim_cfg.device,
            goal_tolerance=GOAL_TOLERANCE,
            robot_index=i, # 로그 출력을 위해 로봇 인덱스 전달
            look_ahead_distance=LOOK_AHEAD_DISTANCE
        )
        waypoint_managers.append(waypoint_manager)
        sim_setup.create_waypoint_markers(robot.get_path(), robot_index=i)
    
    # --- Step 2: 시뮬레이션 초기화 (로봇을 월드에 스폰) ---
    sim_setup.sim.reset()

    # --- Step 3: 컨트롤러 생성 및 초기화 ---
    controllers = []
    for i in range(NUM_ROBOTS):
        robot = robots[i]
        
        if CONTROL_MODE == "ANALYTIC":
            dof_names = robot.articulation.joint_names
            num_dof = len(dof_names)
            left_wheel_indices = [j for j, name in enumerate(dof_names) if "left" in name]
            right_wheel_indices = [j for j, name in enumerate(dof_names) if "right" in name]
            
            controller = AnalyticController(
                num_dof=num_dof,
                wheel_radius=Dingo.WHEEL_RADIUS,
                track_width=Dingo.TRACK_WIDTH,
                left_wheel_indices=left_wheel_indices,
                right_wheel_indices=right_wheel_indices,
                gains=GAINS,
                device=sim_cfg.device,
            )

        elif CONTROL_MODE == "PI":
            # Import the new controller class at the top of the file first!
            # from .controllers import AnalyticController, PIController, IsaacLabIKController

            dof_names = robot.articulation.joint_names
            num_dof = len(dof_names)
            left_wheel_indices = [j for j, name in enumerate(dof_names) if "left" in name]
            right_wheel_indices = [j for j, name in enumerate(dof_names) if "right" in name]
            
            controller = PIController(
                num_dof=num_dof,
                wheel_radius=Dingo.WHEEL_RADIUS,
                track_width=Dingo.TRACK_WIDTH,
                left_wheel_indices=left_wheel_indices,
                right_wheel_indices=right_wheel_indices,
                gains=GAINS,
                device=sim_cfg.device,
            )

        elif CONTROL_MODE == "PID":
            dof_names = robot.articulation.joint_names
            num_dof = len(dof_names)
            left_wheel_indices = [j for j, name in enumerate(dof_names) if "left" in name]
            right_wheel_indices = [j for j, name in enumerate(dof_names) if "right" in name]
            
            controller = PIDController(
                num_dof=num_dof,
                wheel_radius=Dingo.WHEEL_RADIUS,
                track_width=Dingo.TRACK_WIDTH,
                left_wheel_indices=left_wheel_indices,
                right_wheel_indices=right_wheel_indices,
                gains=GAINS,
                device=sim_cfg.device,
            )


        elif CONTROL_MODE == "IK":
            controller = IsaacLabIKController(
                prim_path=robot.prim_path,
                gains=GAINS,
                device=sim_cfg.device,
            )
        else:
            raise ValueError(f"지원하지 않는 컨트롤러 모드입니다: {CONTROL_MODE}")
        
        controllers.append(controller)
        controller.initialize()



    # --- 경로 추적(Path Tracing)을 위한 설정 ---
    draw = _debug_draw.acquire_debug_draw_interface()
    path_history = [[] for _ in range(NUM_ROBOTS)]
    frame_count = 0
    
    # 각 로봇의 경로 색상을 웨이포인트 마커와 동일하게 맞춤
    path_colors = [
        (0.2, 0.6, 1.0, 1.0),  # Blue (RGBA, A=투명도)
        (1.0, 0.6, 0.2, 1.0),  # Orange
        (0.2, 1.0, 0.6, 1.0),  # Teal
        (1.0, 1.0, 0.2, 1.0),  # Yellow
        (0.8, 0.3, 1.0, 1.0),  # Purple
    ]




    # 시뮬레이션 준비 대기
    for _ in range(4):
        sim_setup.sim.step()

    # --- 메인 시뮬레이션 루프 ---
    while simulation_app.is_running():
        sim_setup.sim.step()

        frame_count += 1 # 프레임 카운터 증가

        all_robots_finished = True

        for i in range(NUM_ROBOTS):
            # 이 로봇이 이미 끝났으면 다음 로봇으로 넘어감
            if waypoint_managers[i].is_mission_complete():
                continue
            
            all_robots_finished = False
            
            robot = robots[i]
            waypoint_manager = waypoint_managers[i]
            controller = controllers[i]

            robot.articulation.update(sim_setup.sim.get_physics_dt())

            robot_pos = robot.get_position()

            # --- 경로 기록 ---
            # 10프레임마다 한 번씩 현재 로봇 위치를 경로 기록에 추가
            if frame_count % 10 == 0:
                pos_list = robot_pos.cpu().numpy().tolist()
                path_history[i].append(tuple(pos_list))



            robot_yaw = robot.get_yaw()
            waypoint_manager.update_state(robot_pos)
            target_pos = waypoint_manager.get_target(robot_pos[:2])
            action = controller.compute_action(
                robot_position=robot_pos,
                robot_yaw=robot_yaw,
                target_position=target_pos,
                dt=sim_setup.sim.get_physics_dt() # Pass the simulation time step
            )
            robot.apply_action(action)

        # --- 경로 그리기 ---
        draw.clear_lines() # 이전 프레임의 선들을 지움
        for i, path in enumerate(path_history):
            if len(path) > 1:
                # 1. 시작점 리스트와 끝점 리스트를 분리합니다.
                # 예: path가 [p0, p1, p2, p3] 이면,
                # start_points는 [p0, p1, p2], end_points는 [p1, p2, p3]가 됩니다.
                start_points = path[:-1]
                end_points = path[1:]

                # 2. 각 라인(세그먼트)의 수에 맞게 색상과 크기 리스트를 생성합니다.
                num_lines = len(start_points)
                color = path_colors[i % len(path_colors)]
                colors = [color] * num_lines
                sizes = [2.0] * num_lines

                # 3. 키워드 인자 없이 위치(positional) 인자로 함수를 호출합니다.
                # draw_lines(시작점들, 끝점들, 색상들, 크기들)
                draw.draw_lines(start_points, end_points, colors, sizes)


        if all_robots_finished:
            print("🎉 모든 로봇이 임무를 완료했습니다!")
            break

    # 시뮬레이션 종료
    for robot in robots:
        robot.stop()
    for _ in range(10):
        sim_setup.sim.step()
        
    simulation_app.close()
