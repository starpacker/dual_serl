# PEG Insertion Pipeline using SERL 

This repository contains the necessary pipeline to perform PEG insertion tasks using the Flexiv robotic arm paired with the Robotiq gripper. The project leverages the SERL framework for learning-based control, with communication and control wrappers encapsulated in `serl_robot_infra`.

## Pipeline Overview

To set up and run the PEG insertion task, follow the steps below:

### Prerequisites

Ensure you have all dependencies and environment variables correctly configured before running the pipeline. You will need to activate the environment and run the associated scripts.

### Steps to Run

1. **Activate the environment:**

   ```bash
   bash activate.sh
   ```

2. **Run the Real Learner:**

   ```bash
   bash real_learner.sh
   ```

3. **Run the Real Actor:**

   ```bash
   bash real_actor.sh
   ```

### Folder Structure

- **`serl_robot_infra/`:** Contains communication modules and wrappers for interacting with the Flexiv robotic arm and Robotiq gripper.
- **`serl_launcher/`:** Includes necessary environmental settings and configurations required to launch the pipeline.

### Hardware

- **Robot Arm:** [Flexiv Robotic Arm](https://www.flexiv.com)
- **Gripper:** [Robotiq Gripper](https://robotiq.com)

Make sure your hardware is correctly set up and connected before running the pipeline.
