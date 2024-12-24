# CITO for manipulation

## Setup Environment

1. Create a new Conda environment:
    ```bash
    conda create -n CITO python=3.12
    ```
   
2. Activate the environment:
    ```bash
    conda activate CITO
    ```

3. Install necessary dependencies:

    ```bash
    conda install boost -c conda-forge
    ```

    ```bash
    conda install cmake=3.23 -c conda-forge
    ```

    ```bash
    conda install matplotlib imageio mim-solvers mujoco meshcat-python=0.3.0 -c conda-forge
    ```

    
4. Build and install:
   ```bash
   git clone --recursive https://github.com/JianghanZHang/CITO.git 
   ```
   ```bash
   mkdir build
   ```
   ```bash
   cd build
   ```
   ```bash
   cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release
   ```
   ```bash
   make [-j8]
   ```
   ```bash
   sudo make install 
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
