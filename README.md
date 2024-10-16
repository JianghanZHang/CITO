# CITO for manipulation

## Setup Environment

1. Create a new Conda environment:
    ```bash
    conda create -n CITO
    ```
   
2. Activate the environment:
    ```bash
    conda activate CITO
    ```

3. Install necessary dependencies:
    ```bash
    conda install pinocchio crocoddyl matplotlib imageio meshcat-python -c conda-forge
    ```

4. Build and install `mim_solver-devel`:
    - Clone and install the development branch from the repository: [mim_solvers (devel)](https://github.com/machines-in-motion/mim_solvers/tree/devel)

5. Install MuJoCo (not available in Conda):
    ```bash
    pip install mujoco==3.2.3
    ```

## Running an Example

To run a minimal example, use the following command:

```bash
python examples/go2_walking_MJ.py [CSQP]
```
## State Conventions:
    For quaternions, refer to https://github.com/clemense/quaternion-conventions
    State:
        Piniocchio:
                q = [global_base_position, global_base_quaternion, joint_positions]
                v = [local_base_velocity_linear, local_base_velocity_angular, joint_velocities]
        Mujoco:
                q = [global_base_position, global_base_quaternion, joint_positions]
                v = [global_base_velocity_linear, local_base_velocity_angular, joint_velocities]