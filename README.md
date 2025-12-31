# EFMNode

EFMNode is a robot control node that integrates vision-language models (VLM) with robot action inference for real-time robot control via ROS2.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Model Weights and Configuration](#model-weights-and-configuration)
- [Running and Data Recording](#running-and-data-recording)
- [Troubleshooting](#troubleshooting)

## Environment Setup

### Prerequisites
- CUDA-capable GPU (RTX 4090 with CUDA driver over 12.8 is recommended)
- ROS2 (Humble or later)
- Python 3.8+

### Installation Steps

1. **Clone the EFMNode repository**:
```bash
git clone https://github.com/OpenGalaxea/EFMNode.git
cd EFMNode
```

2. **Clone and install the GalaxeaFM repository** (if not already done):
```bash
git clone https://github.com/OpenGalaxea/GalaxeaFM.git
cd GalaxeaFM
uv sync --index-strategy unsafe-best-match
uv pip install -e .
source .venv/bin/activate
cd ..
```

## Model Weights and Configuration

### Download Model Weights

1. Download the model checkpoint directory from the provided source
2. Extract it to your desired location (e.g., `/path/to/model/checkpoint/`)

The checkpoint directory should contain:
- `config.yaml` - Model configuration file
- `dataset_stats.json` - Dataset statistics for preprocessing
- Model weight files (`.pth`, `.pt`, or TensorRT engine files)

**Note for TensorRT users**: When using TensorRT (`use_trt = true`), the TensorRT plugin `gemma_rmsnorm.so` should be placed in `plugins/lib/` directory. The plugin path is automatically resolved relative to the project root.

### Configure `config.toml`

Edit `config.toml` in the project root to match your setup:

```toml
[robot]
hardware = "R1_LITE"  # or "R1_PRO"

[basic]
control_frequency = 15.0  # Control frequency in Hz
step_mode = "sync"        # "sync" or "async"
action_steps = 32         # Number of action steps per inference

[model]
ckpt_dir = "/path/to/ckpt/dir"  # Path to model checkpoint directory
use_trt = false                 # Set to true to use TensorRT acceleration
processor = "default"           # Processor type

[instruction]
use_vlm = true                      # Enable VLM-based instructions
bbox_as_instruction = false         # Use bounding boxes as instructions
image_condition_lang_prefix = true  # Use image condition with language prefix
pp_lower_half = false               # Post-process lower half
image_as_condition = true           # Use image as condition
```

**Important Configuration Notes:**
- Set `ckpt_dir` to the absolute path of your model checkpoint directory
- Set `use_trt = true` if you have TensorRT engines available
- Adjust `control_frequency` based on your hardware capabilities
- Modify `hardware` to match your robot model

## Running and Data Recording

### Running the Node

1. **Ensure ROS2 is sourced**:
```bash
source /opt/ros/humble/setup.bash  # Adjust path for your ROS2 installation
```

2. **Run the node**:
```bash
./scripts/run.sh
```

The script `run.sh` automatically:
- Sets up environment variables (HF_ENDPOINT, PYTHONPATH, LD_LIBRARY_PATH)
- Configures logging
- Runs the main script

### Data Recording

To record ROS2 topics for later analysis or replay:

```bash
./scripts/record.sh <output_bag_directory>
```

Example:
```bash
./scripts/record.sh /path/to/recordings/my_recording
```

The recording script captures:
- Camera topics (head depth, left/right wrist cameras)
- Feedback topics (grippers, arms, torso, chassis)
- Control topics (arm, gripper, chassis, torso control)
- Motion target topics
- TF transforms
- Progress indicators

**Note:** If the output directory already exists, it will be deleted before recording starts.

### Running with Custom Configuration

You can modify `config.toml` and restart the node to apply changes. The node will:
1. Load the configuration from `config.toml`
2. Load model configuration from `{ckpt_dir}/config.yaml`
3. Initialize the inference engine and processor
4. Connect to ROS2 topics
5. Start the control loop

## Troubleshooting

### Common Issues

#### 1. **"No observation" messages**
- **Cause**: ROS2 topics are not publishing or not connected
- **Solution**: 
  - Verify ROS2 is running: `ros2 topic list`
  - Check if required topics are available
  - Ensure the robot hardware is connected and publishing

#### 2. **Model loading errors**
- **Cause**: Incorrect checkpoint path or missing files
- **Solution**:
  - Verify `ckpt_dir` in `config.toml` is correct
  - Ensure `config.yaml` and `dataset_stats.json` exist in the checkpoint directory
  - Check file permissions
