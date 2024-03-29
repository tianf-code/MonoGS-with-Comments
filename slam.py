import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)    # Create a CUDA event to record the start time
        end = torch.cuda.Event(enable_timing=True)  # Create a CUDA event to record the end time

        start.record()  # Record start time

        self.config = config
        self.save_dir = save_dir

        # 3DGS parameters
        model_params = munchify(config["model_params"]) # model parameters
        opt_params = munchify(config["opt_params"]) # optimization parameters
        pipeline_params = munchify(config["pipeline_params"])   # pipeline parameters
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        # SLAM parameters
        self.live_mode = self.config["Dataset"]["type"] == "realsense"  # if use realsense, then it's live mode
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0   # sh parameters

        # Initialize 3D Gaussian Splatting Model
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)

        # Load dataset
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)   # Set optimization parameters to Gaussian models
        bg_color = [0, 0, 0]    # Set background color: Black
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")    # Put background color tensor on cuda

        # Create two multi-process queues for communication between the front end and back end
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        # Create two GUI multi-process queues
        # According to whether to use the GUI, choose to use mp.Queue() or FakeQueue() to create a fake queue object
        # If the program needs to communicate with the GUI process, use a real multi-process queue;
        # Use a fake queue if not required to avoid throwing exceptions without a GUI.
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        # Update configs
        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        # Create front-end and back-end instances and set parameters
        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        # Background color, pipeline parameters, frontend queue and backend queue are shared between the front end and back end
        # Dataset is in frontend
        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        # GUI is in frontend
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        # Gaussians and optimization are in back end
        self.backend.gaussians = self.gaussians
        self.backend.cameras_extent = 6.0
        self.backend.opt_params = self.opt_params
        self.backend.background = self.background
        self.backend.pipeline_params = self.pipeline_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        # Start SLAM
        backend_process = mp.Process(target=self.backend.run)   # Create a background process and set target function as self.backend.run()
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)
        backend_process.start() # Start the backend

        self.frontend.run() # Start the frontend
        backend_queue.put(["pause"])

        # SLAM ends, record end time
        end.record()
        Log("Frontend stopped")

        # Evaluation
        torch.cuda.synchronize()    # Synchronize the CPU and GPU to ensure that calculations on the GPU are completed
        N_frames = len(self.frontend.cameras)   # Number of camera frames processed by the front end
        FPS = N_frames / (start.elapsed_time(end) * 0.001)  # Calculate the frame rate.
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians    # Gaussians after front-end processing
            kf_indices = self.frontend.kf_indices   # keyframes indices

            # Calculate RMSE Absolute Trajectory Error (ATE)
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            # Evaluate rendering results (Before optimization)
            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # Then, prepare to evalute the rendering results after optimization
            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():   # Empty frontend queue
                frontend_queue.get()
            backend_queue.put(["color_refinement"])

            # Wait in an infinite loop to obtain synchronized signal from the front-end queue
            # and get the optimized Gaussians from the back-end queue
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians  # Get optimized Gaussians
                    del data
                    break

            # Evaluate rendering results (After optimization)
            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)    # Save optimized Gaussians

        # Close frontend queue
        frontend_queue.close()
        frontend_queue.join_thread()

        # End backend and GUI process
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            q_main2vis.close()
            q_main2vis.join_thread()
            q_vis2main.close()
            q_vis2main.join_thread()
            gui_process.terminate()
            Log("GUI Stopped and joined the main thread")
        backend_queue.put(["stop"])
        backend_queue.close()
        backend_queue.join_thread()
        backend_process.join()  # Wait for background process to complete
        Log("Backend stopped and joined the main thread")

    # Fake function
    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)   # Load the config file
    parser.add_argument("--eval", action="store_true")  # Load the evaluation parameter (whether to run in evaluation mode)

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")    # start a method with torch.multiprocessing

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:   # If the --eval argument is passed
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True    # Results will be saved
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True  # Set to True to perform rendering assessment
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])  # Create a directory to save results, the path is in the config file
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # Get the current date and time
        path = config["Dataset"]["dataset_path"].split("/") # Split the dataset path by "/"
        save_dir = os.path.join( # Joint the dataset path with the current time as the directory for saving results.
            config["Results"]["save_dir"], path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0] # Remove the extension from the filename (args.config)
        config["Results"]["save_dir"] = save_dir    # Update the directory path in configs for saving results
        mkdir_p(save_dir)   # Make the directory to save results
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)

        # Initialize WandB (Weights and Biases) environment to track experiment results.
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        # Two metrics are defined to track the progress of the experiment on WandB.
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)  # Initialize the SLAM instance, pass the configuration and save_dir path.

    slam.run()  # In fact, all SLAM processes are completed in init(), and run() has no real role
    wandb.finish()

    # All done
    Log("Done.")