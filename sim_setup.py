"""
simulation_setup.py

Isaac Sim의 시뮬레이션 월드 환경을 구성하는 SimulationSetup 클래스를 정의합니다.
- 물리 환경 (중력 등)
- 기본 지형 (바닥) 및 시각 효과 (그리드)
- 조명 설정
- 웨이포인트 시각적 마커 생성
"""

from typing import List, Tuple

# Isaac Lab / Core
from omni.isaac.lab.sim import SimulationContext, SimulationCfg
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.utils.stage import get_current_stage

# USD / PhysX / Lux
from pxr import Usd, UsdLux, UsdGeom, Gf, UsdPhysics, PhysxSchema


class SimulationSetup:
    """
    Isaac Sim의 시뮬레이션 월드와 관련된 모든 설정을 관리합니다.
    """

    def __init__(self, sim_cfg: SimulationCfg):
        """
        SimulationSetup을 초기화합니다.

        Args:
            sim_cfg (SimulationCfg): 시뮬레이션 설정을 담은 데이터 클래스.
        """
        self.sim = SimulationContext(sim_cfg)
        self.stage = get_current_stage()

    def setup_scene(self):
        """
        장면의 기본 요소들을 설정합니다.
        물리, 바닥, 조명 등을 포함합니다.
        """
        print("🌍 시뮬레이션 장면 구성을 시작합니다...")
        self._setup_stage_settings()
        self._setup_physics()
        self._create_ground_plane()
        self._create_lights()
        print("✅ 장면 구성이 완료되었습니다.")

    def _setup_stage_settings(self):
        """스테이지의 기본 단위를 설정합니다 (Z-up, meters)."""
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)

    def _setup_physics(self):
        """물리 장면을 정의하고 중력을 설정합니다."""
        phys_scene_path = "/physicsScene"
        if not self.stage.GetPrimAtPath(phys_scene_path):
            phys_scene = UsdPhysics.Scene.Define(self.stage, phys_scene_path)
            phys_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
            phys_scene.CreateGravityMagnitudeAttr(9.81)
            # 안정적인 물리 시뮬레이션을 위한 설정
            PhysxSchema.PhysxSceneAPI.Apply(phys_scene.GetPrim()).CreateEnableStabilizationAttr(True)

    def _create_ground_plane(self, grid_size: float = 10.0):
        """충돌이 적용된 바닥 평면과 시각적 그리드를 생성합니다."""
        # 바닥 평면 (거대한 큐브)
        ground_path = "/World/Ground"
        prim_utils.create_prim(
            prim_path=ground_path,
            prim_type="Cube",
            scale=(2 * grid_size * 10, 2 * grid_size * 10, 0.2),
            position=(0.0, 0.0, -0.1),
            attributes={"size": 1.0},
        )
        ground_prim = self.stage.GetPrimAtPath(ground_path)
        UsdPhysics.CollisionAPI.Apply(ground_prim)

        gprim_ground = UsdGeom.Gprim(ground_prim)
        dark_color = (0.1, 0.1, 0.1)
        gprim_ground.CreateDisplayColorAttr().Set([Gf.Vec3f(*dark_color)])

        # 바닥 그리드 (시각적 효과)
        grid_path = "/World/Ground/Grid"
        prim_utils.create_prim(grid_path, "Xform")
        major_color = (0.25, 0.25, 0.25)
        
        # 간단한 그리드 표현
        for i in range(-int(grid_size), int(grid_size) + 1):
            # --- 안전한 이름 생성 (이전과 동일) ---
            sign = "p" if i >= 0 else "n"
            safe_name_x = f"line_x_{sign}{abs(i):02d}"
            safe_name_y = f"line_y_{sign}{abs(i):02d}"
            
            # --- X축 라인 ---
            pos_x = (0, i, 0.002)
            scale_x = (2 * grid_size, 0.0001, 0.00001)
            prim_path_x = f"{grid_path}/{safe_name_x}"
            
            # 1. 속성 없이 프리미티브 먼저 생성
            prim_utils.create_prim(prim_path_x, "Cube", position=pos_x, scale=scale_x)
            
            # 2. 생성된 프리미티브를 가져와서 색상 속성을 직접 생성하고 설정
            prim_x = self.stage.GetPrimAtPath(prim_path_x)
            gprim_x = UsdGeom.Gprim(prim_x)
            gprim_x.CreateDisplayColorAttr().Set([Gf.Vec3f(*major_color)])
            
            # --- Y축 라인 ---
            pos_y = (i, 0, 0.002)
            scale_y = (0.0001, 2 * grid_size, 0.00001)
            prim_path_y = f"{grid_path}/{safe_name_y}"

            # 1. 속성 없이 프리미티브 먼저 생성
            prim_utils.create_prim(prim_path_y, "Cube", position=pos_y, scale=scale_y)

            # 2. 생성된 프리미티브를 가져와서 색상 속성을 직접 생성하고 설정
            prim_y = self.stage.GetPrimAtPath(prim_path_y)
            gprim_y = UsdGeom.Gprim(prim_y)
            gprim_y.CreateDisplayColorAttr().Set([Gf.Vec3f(*major_color)])
    
    def _create_lights(self):
        """장면에 그림자 없는 부드러운 조명을 설정합니다."""
        lights_path = "/World/Lights"
        prim_utils.create_prim(lights_path, "Xform")

        # 하늘에서 비추는 메인 조명 (그림자 없음)
        rect_light = UsdLux.RectLight.Define(self.stage, f"{lights_path}/RectLight")
        rect_light.CreateIntensityAttr(750.0)
        rect_light.CreateWidthAttr(200.0)
        rect_light.CreateHeightAttr(200.0)
        UsdGeom.XformCommonAPI(rect_light).SetTranslate(Gf.Vec3d(0.0, 0.0, 20.0))
        UsdLux.ShadowAPI.Apply(rect_light.GetPrim()).CreateShadowEnableAttr(False)
        
        # 전체적인 환경광을 위한 돔 라이트
        dome_light = UsdLux.DomeLight.Define(self.stage, f"{lights_path}/DomeLight")
        dome_light.CreateIntensityAttr(400.0)

   

   

    def create_waypoint_markers(self, waypoints: List[Tuple[float, float]], robot_index: int, radius: float = 0.08, z_offset: float = 0.015):
        """
        주어진 웨이포인트 목록을 시뮬레이션 월드에 시각적으로 표시합니다.
        한 로봇의 모든 마커는 동일한 색상으로 표시됩니다.
        """
        # 각 로봇의 경로를 구별하기 위한 색상 팔레트
        path_colors = [
            (0.2, 0.6, 1.0),  # Blue
            (1.0, 0.6, 0.2),  # Orange
            (0.2, 1.0, 0.6),  # Teal
            (1.0, 1.0, 0.2),  # Yellow
            (0.8, 0.3, 1.0),  # Purple
        ]

        markers_path = "/World/WaypointMarkers"
        if not self.stage.GetPrimAtPath(markers_path):
            prim_utils.create_prim(markers_path, "Xform")
        
        # 로봇 인덱스를 기반으로 이 경로에 사용할 단일 색상을 선택합니다.
        path_color = path_colors[robot_index % len(path_colors)]
        
        for i, (x, y) in enumerate(waypoints):
            # 고유한 마커 경로 생성
            marker_path = f"{markers_path}/robot_{robot_index}_wp_{i:03d}"
            sphere = UsdGeom.Sphere.Define(self.stage, marker_path)
            sphere.CreateRadiusAttr(radius)
            
            # Gprim API를 사용하여 선택된 경로 색상을 설정합니다.
            gprim = UsdGeom.Gprim(sphere.GetPrim())
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*path_color)])
            
            # XformCommonAPI를 사용하여 위치 설정
            xform_api = UsdGeom.XformCommonAPI(sphere)
            xform_api.SetTranslate(Gf.Vec3d(float(x), float(y), z_offset))

        print(f"📍 [Robot {robot_index}] 웨이포인트 마커 {len(waypoints)}개를 생성했습니다.")