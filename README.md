# LLM Service Setup Guide

This repository provides instructions to set up and run the LLM service inside a k8s pod. Follow the steps below to install dependencies, configure the environment, and build the necessary services.

## Installation Steps

### 1. Install Python Virtual Environment

```bash
sudo apt update --fix-missing
sudo apt install python3.10-venv
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv LLM
source LLM/bin/activate
```

### 3. Install Dependencies (One by One to Avoid Failures)

```bash
pip3 install jinja2
pip3 install pyyaml typeguard
pip3 install torch torchvision torchaudio
pip3 install opencv-python
pip3 install -U huggingface_hub
```

### 4. Download Pretrained Models

```bash
mkdir pretrained
cd pretrained
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-8B --local-dir InternVL2-8B
cd ..
```

### 5. Install Additional Dependencies

```bash
pip3 install einops wheel timm
pip3 install flash_attn
pip3 install decord
pip3 install transformers
pip3 install accelerate
pip3 install sentencepiece
python3 -m pip install -U scikit-image
```

### 6. Build ROS2 Service

```bash
cd colcon_ws/src
git clone https://github.com/Sisqui/LLM_service.git
GIT_LFS_SKIP_SMUDGE=1 git clone -b humble-devel https://github.com/Alpaca-zip/ultralytics_ros.git
rosdep install -r -y -i --from-paths .
```

### 7. Install ROS2 Dependencies

```bash
python3 -m pip install -r ultralytics_ros/requirements.txt
pip3 install empy==3.3.4
pip3 install catkin_pkg lark
pip3 install numpy==1.26.4
```

### 8. Build and Source ROS2 Packages

```bash
cd ..
colcon build --packages-select llm image_pose_pub ultralytics_ros
source install/setup.bash
```

## Usage

Once all dependencies are installed and ROS2 services are built, you can start using the LLM service within your ROS2 environment. Ensure your virtual environment is activated before running any commands.

### rap

```bash
# Terminal 1
zenoh-bridge-ros2dds -c zenoh-config-drone.json5
# Terminal 2: 
ros2 launch image_pose_pub image_pose_launch.py
# Terminal 3: 
ros2 launch ultralytics_ros tracker.launch.xml debug:=true
#Terminal 4: 
python3 ~/colcon_ws/src/LLM_service/leakage/scripts/ros2service-posture.py
#Terminal 5: 
python3 ~/colcon_ws/src/LLM_service/leakage/scripts/ros2client-posture-v4.py
```

### Drone

```bash
# Terminal 1:
ros2 launch zed_wrapper zed_camera.launch.py 
# Terminal 2:
zenoh-bridge-ros2dds -c zenoh-config-drone.json5
# Terminal 3: 
python3 ~/ros2_ws/src/LLM_service/leakage/scripts/tf_transform_marker.py 
```
