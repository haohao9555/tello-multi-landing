cmake_minimum_required(VERSION 3.8)
project(tello_sim_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config models worlds DESTINATION share/${PROJECT_NAME})

# source install/setup.bash 时自动把 models/ 加进 GZ_SIM_RESOURCE_PATH
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in")

ament_package()
