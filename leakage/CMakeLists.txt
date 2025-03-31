cmake_minimum_required(VERSION 3.8)
project(llm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_geometry REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}

  "srv/CheckImage.srv"
  "srv/CheckPosture.srv"
  "msg/PoseWithCompressedImage.msg"
  "msg/PoseWithString.msg"
  DEPENDENCIES sensor_msgs
)

install(PROGRAMS
  scripts/ros2client-posture-summit.py
  scripts/ros2service-posture.py
  scripts/tf_transform_marker.py
  scripts/ros2service.py
  scripts/ros2client-leakage.py
  scripts/ros2client-summit.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
