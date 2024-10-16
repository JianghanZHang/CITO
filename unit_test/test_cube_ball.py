from robots.robot_env import create_cube_pusher_env, create_cube_pusher_env_MJ
import mujoco 
import pinocchio as pin
pin_env = create_cube_pusher_env()
pin_model = pin_env["rmodel"]
pin_data = pin_model.createData()
print(pin_model, "\n")
mj_env = create_cube_pusher_env_MJ()
mj_model = mj_env["mj_model"]
mj_data = mujoco.MjData(mj_model)

import pdb; pdb.set_trace()