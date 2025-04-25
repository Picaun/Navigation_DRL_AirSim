# Navigation_DRL_UAV 项目说明文档

## 项目概述

Navigation_DRL_UAV  是一个基于深度强化学习（DRL）的无人机导航平台，用于在复杂未知环境中训练无人机导航策略。该平台基于 AirSim 和 Stable-Baselines3，包含多旋翼和固定翼无人机的运动学模型，并提供多种 UE4 环境用于训练和测试。

## Quick Start

### 环境配置

1. 如果使用conda配置
最好使用conda虚拟环境配置，避免环境混杂
   1. `conda create -n DRL_AirSim python=3.8.20`
   2. `conda activate DRL_AirSim`激活DRL_AirSim环境

2. Install CUDA and PyTorch (Win10 or Win11)

- Download [CUDA11.6](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)（先不装，如果有cuda兼容问题再装此版本）
- `pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
- You can use `tools/test/torch_gpu_cpu_test.py` to test your PyTorch and CUDA.

3. Install Requirements

   1. `pip install numpy==1.24.4 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`
   2. `pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`
4. Install gym_env

   1. `pip install -e ./gym_env`

5. Install customized stable-baselines3

   1.  `pip install -e ./stable-baselines3`

### 启动训练

   1. 运行“Navigation_DRL_UAV\UE4\AirSimNH\WindowsNoEditor\AirSimNH.exe”虚幻4仿真环境（自行下载）
   2. 在“Choose Vehicle”弹窗选择否
    回到终端：
   4. `python ./scripts/start_train_with_plot.py`
   
## Configs文件说明

This repo using config file to control training conditions.

Now we provide 3 training envrionment and 3 dynamics.

**env_name**

* SimpleAvoid

  * This is a custom UE4 environment used for simple obstacle avoidance test. You can download it from [google drive](https://drive.google.com/file/d/1QgkZY5-GXRr93QTV-s2d2OCoVSndADAM/view?usp=sharing).

  <p align="center">
    <img src="resources/env_maps/simple_world_1.png" width = "350" height = "200"/>
    <img src="resources/env_maps/simple_world_45.png" width = "350" height = "200"/>
  </p>
* City_400_400

  * A custom UE4 environment used for fixedwing obstacle avoidance test. You can also get it from [google drive](https://drive.google.com/file/d/1vdT8cP2n_jTk1MdShGwdf1OFPC_YOmTr/view?usp=sharing)

  <p align="center">
    <img src="resources/env_maps/city_400.png" width = "350" height = "200"/>
    <img src="resources/env_maps/city_400_1.png" width = "350" height = "200"/>
  </p>
* Random obstacles

  * Some envs with random obstacles. Contributed by [Chris-cch](https://github.com/Chris-cch). You can download [here](https://mailnwpueducn-my.sharepoint.com/personal/chenchanghao_mail_nwpu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchenchanghao%5Fmail%5Fnwpu%5Fedu%5Fcn%2FDocuments%2Fnew%5Fenv&ga=1).

  <p align="center">
    <img src="resources/env_maps/random_1.png" width = "350" height = "200"/>
    <img src="resources/env_maps/random_2.png" width = "350" height = "200"/>
  </p>
* Other Airsim build in envrionment (AirSimNH and CityEnviron):

  <p align="center">
    <img src="resources\env_maps\NH.png" width = "350" height = "200"/>
    <img src="resources\env_maps\city.png" width = "350" height = "200"/>
  </p>

**dynamic_name**

* SimpleMultirotor
* Multirotor
* SimpleFixedwing

## 目录结构及功能说明

### 根目录文件

- **requirements.yml**：项目依赖的 Python 包和环境配置，使用 conda 创建环境。
- **README.md**：项目的详细说明文档，包含安装、配置和使用说明。

### 主要目录

#### `/gym_env` - 强化学习环境

这是项目的核心组件，提供了与 AirSim 交互的 Gym 环境。

- **gym_env/gym_env/envs/airsim_env.py**：主要的环境类 `AirsimGymEnv`，继承自 `gym.Env`，实现了强化学习环境的接口，包含状态观测、动作执行、奖励计算等功能。
- **gym_env/gym_env/envs/dynamics/**：包含不同类型的无人机动力学模型
  - **multirotor_simple.py**：简化的多旋翼无人机动力学模型
  - **multirotor_airsim.py**：基于 AirSim 的多旋翼无人机动力学模型
  - **fixedwing_simple.py**：简化的固定翼无人机动力学模型
- **gym_env/setup.py**：用于将 gym_env 安装为 Python 包

#### `/UE4` - 虚拟环境

包含用于训练和测试的虚幻引擎环境：

- **SimpleAvoid/**：简单的障碍物避障测试环境
- **AirSimNH/**：AirSim 内置的 NeighborhoodEnvironment 环境
- **AirSimNH.zip**：AirSim 内置的 NeighborhoodEnvironment 环境压缩包
- **ZhangJiajie.zip**：张家界景观环境压缩包
- **SimpleAvoid.rar**：简单避障环境压缩包

#### `/scripts` - 训练和评估脚本

包含用于训练和评估模型的脚本：

- **start_train_with_plot.py**：启动训练过程并实时可视化训练情况
- **start_evaluate_with_plot.py**：评估训练好的模型并可视化结果
- **train.py**：主要的训练逻辑
- **evaluation.py**：评估模型的逻辑
- **utils/**：各种辅助功能和工具函数

#### `/configs` - 配置文件

包含不同训练场景的配置文件：

- **config_NH_center_Multirotor_3D.ini**：NeighborhoodEnvironment 中心区域的多旋翼训练配置
- **config_fixedwing.ini**：固定翼无人机的训练配置
- **config_Maze_SimpleMultirotor_2D.ini**：迷宫环境中简化多旋翼的训练配置
- **config_Trees_SimpleMultirotor.ini**：树林环境中简化多旋翼的训练配置

#### `/stable-baselines3` - 强化学习库

定制版的 Stable-Baselines3 库，用于实现深度强化学习算法：

- **stable_baselines3/**：包含 PPO、SAC、TD3 等强化学习算法的实现
- **tests/**：测试代码
- **scripts/**：辅助脚本
- **docs/**：文档

#### `/airsim_settings` - AirSim 配置

AirSim 模拟器的配置文件，用于设置模拟环境参数。

#### `/logs`、`/logs_eval`、`/logs_save` - 日志和模型保存

- **logs/**：训练过程的日志
- **logs_eval/**：评估过程的日志
- **logs_save/**：保存的模型和训练结果

#### `/resources` - 资源文件

包含项目使用的图像、图表等资源文件。

#### `/tools` - 工具脚本

包含各种辅助工具，如 PyTorch GPU 测试脚本等。

## 核心组件交互关系

- **AirsimGymEnv**：作为核心环境类，与 AirSim 交互获取感知信息，将动作发送给无人机
- **动力学模型**：实现不同类型无人机的运动学特性，由环境类调用
- **训练算法**：使用 Stable-Baselines3 中的算法（PPO、SAC、TD3）进行训练
- **可视化界面**：使用 PyQt 实现的实时训练和评估可视化界面