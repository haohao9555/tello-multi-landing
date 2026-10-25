"""Sim.launch."""
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
                              description='true: run the simulation server without a GUI'),
        DeclareLaunchArgument('world', default_value='tello_land.sdf',
                              description='World filename under tello_sim_description/worlds/'),
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
