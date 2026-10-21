"""一键启动 Tello 仿真：Gazebo + ros_gz_bridge + 接口节点。

  ros2 launch tello_sim_bringup sim.launch.py                 # 带 GUI
  ros2 launch tello_sim_bringup sim.launch.py headless:=true  # 无界面
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_gazebo(context):
    desc = get_package_share_directory('tello_sim_description')
    world = os.path.join(desc, 'worlds', LaunchConfiguration('world').perform(context))
    if LaunchConfiguration('headless').perform(context).lower() == 'true':
        gz_args = f'-s -r {world}'
    else:
        gz_args = f'-r --gui-config {os.path.join(desc, "config", "gui.config")} {world}'
    gz_sim_launch = os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch),
        launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items())]


def generate_launch_description():
    desc = get_package_share_directory('tello_sim_description')
    bringup = get_package_share_directory('tello_sim_bringup')
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false',
                              description='true：只跑仿真服务器，不开 GUI'),
        DeclareLaunchArgument('world', default_value='tello_land.sdf',
                              description='tello_sim_description/worlds/ 下的文件名'),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(desc, 'models')),
        OpaqueFunction(function=launch_gazebo),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='tello_bridge',
             parameters=[{'config_file': os.path.join(bringup, 'config', 'bridge.yaml'),
                          'use_sim_time': True}],
             output='screen'),
        Node(package='tello_sim_interface', executable='interface_node',
             name='tello_sim_interface',
             parameters=[os.path.join(bringup, 'config', 'tello_params.yaml')],
             output='screen'),
    ])
