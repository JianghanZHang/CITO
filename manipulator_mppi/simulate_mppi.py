import argparse
import faulthandler

# from control.controllers.mppi_reaching import reaching_MPPI
from control.controllers.mppi_manipulation import manipulation_MPPI
from control.controllers.mppi_reaching import reaching_MPPI

# Adjust these imports to match where you have placed the trifinger code:
from interface.simulator import Simulator
from utils.tasks import get_task


def main(task):
    # ---------------------------
    # Simulation and Controller Parameters
    # ---------------------------
    T = 2000  # total steps, e.g. 20 seconds if dt=0.01

    # ---------------------------
    # Get trifinger-specific task data
    # ---------------------------
    # For example, tasks might be { "reaching": {"sim_path": "..."} }
    task_data = get_task(task)
    sim_path = task_data["sim_path"]  # path to your trifinger xml

    # ---------------------------
    # Initialize MPPI and simulator
    # ---------------------------

    if task == "reaching":
        agent = reaching_MPPI(task=task)

    else:
        agent = manipulation_MPPI(task=task)

    VIEWER = True
    SIMULATION_STEP = 0.01
    CTRL_UPDATE_RATE = 100     # control update frequency
    # Soft contact model parameters
    TIMECONST = 0.02
    DAMPINGRATIO = 1.0
    simulator = Simulator(
        agent=agent,
        viewer=VIEWER,
        T=T,
        dt=SIMULATION_STEP,
        timeconst=TIMECONST,
        dampingratio=DAMPINGRATIO,
        model_path=sim_path,
        ctrl_rate=CTRL_UPDATE_RATE
    )

    # ---------------------------
    # Run simulation + plotting
    # ---------------------------
    simulator.run()
    simulator.plot_trajectory()


if __name__ == "__main__":

    faulthandler.enable()

    # Example trifinger tasks:


    VALID_TASKS = ["reaching", "cube_manipulation", "cube_planar_push", "bowl_planar_push", 
                   "mug_planar_push", "can_planar_push", "lightbulb_planar_push", "flashlight_planar_push","rubberduck_planar_push",
                   "elephant_planar_push", "torus_planar_push", "airplane_planar_push", "camera_planar_push", "bunny_planar_push",
                   "teapot_planar_push", "banana_planar_push"]

    parser = argparse.ArgumentParser(description="Run trifinger MPPI simulation.")
    parser.add_argument('--task',
                        type=str,
                        required=True,
                        choices=VALID_TASKS,
                        help=f"Name of the task. Must be one of: {VALID_TASKS}")
    args = parser.parse_args()

    main(args.task)
