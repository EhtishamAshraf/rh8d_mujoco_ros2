cmake_minimum_required(VERSION 3.8)
project(rh8d_mujoco_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HandCommand.msg"
  "msg/HandState.msg"
  "msg/FingertipForces.msg"
  "msg/HandContacts.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()