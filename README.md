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
    conda install pinocchio crocoddyl matplotlib imageio mim-solvers meshcat-python=0.3.0 mujoco==3.2.0 -c conda-forge
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
