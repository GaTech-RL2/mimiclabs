"""
Script to initialize an environment multiple times and capture initial agentview images as a video.

This script takes a task_suite and task_name, creates a MimicLabs environment using robosuite.make(),
resets it multiple times, and stores the first image capture ("agentview_image" key in observations) 
from each reset as frames in a video.

Example usage:

    For a MimicLabs task:

    python scripts/capture_reset_frames.py \
        --task_suite_name example_suite \
        --task_name example_task \
        --num_resets 20 \
        --output_video reset_frames.mp4 \
        --fps 10
    
    For Robosuite tasks:

    python scripts/capture_reset_frames.py \
        --task_name Lift \
        --num_resets 20 \
        --output_video reset_frames.mp4 \
        --fps 10
"""

import os
import argparse
import numpy as np
import imageio
from tqdm import tqdm

import robosuite
import mimiclabs
import mimiclabs.mimiclabs.envs.bddl_utils as BDDLUtils
from mimiclabs.mimiclabs.envs.bddl_base_domain import TASK_MAPPING


def capture_reset_frames(args):
    """
    Initialize environment multiple times and capture initial agentview images.
    """
    print(
        f"Setting up environment for task suite: {args.task_suite_name}, task: {args.task_name}"
    )

    # Setup environment configuration
    if args.task_suite_name is not None:
        # MimicLabs task suite
        bddl_file_name = os.path.join(
            mimiclabs.__path__[0],
            "mimiclabs",
            "task_suites",
            args.task_suite_name,
            f"{args.task_name}.bddl",
        )

        if not os.path.exists(bddl_file_name):
            raise FileNotFoundError(f"BDDL file not found: {bddl_file_name}")

        # Parse the BDDL to get environment class
        parsed_problem = BDDLUtils.robosuite_parse_problem(bddl_file_name)
        env_class = TASK_MAPPING[parsed_problem["problem_name"]]
        env_name = env_class.__name__

        print(f"Found environment class: {env_name}")
        print(f"BDDL file: {bddl_file_name}")

        # Environment kwargs for MimicLabs tasks
        env_kwargs = {
            "bddl_file_name": bddl_file_name,
            "robots": args.robots,
            "has_renderer": False,  # No on-screen rendering
            "has_offscreen_renderer": True,  # Enable camera capture
            "camera_names": ["agentview"],  # We want agentview images
            "camera_heights": args.image_height,
            "camera_widths": args.image_width,
            "control_freq": 20,
            "horizon": 1000,
        }
    else:
        # Standard robosuite environment
        env_name = args.task_name
        env_kwargs = {
            "robots": args.robots,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "camera_names": ["agentview"],
            "camera_heights": args.image_height,
            "camera_widths": args.image_width,
            "control_freq": 20,
            "horizon": 1000,
        }

    print(f"Creating environment: {env_name}")
    print(f"Environment kwargs: {env_kwargs}")

    # Create environment
    try:
        env = robosuite.make(env_name, **env_kwargs)
        print("Environment created successfully!")
    except Exception as e:
        print(f"Error creating environment: {e}")
        raise

    # Collect initial frames
    print(f"Collecting {args.num_resets} initial frames...")
    frames = []

    for i in tqdm(range(args.num_resets), desc="Capturing frames"):
        try:
            # Reset environment
            obs = env.reset()

            # Extract agentview image
            if "agentview_image" in obs:
                frame = obs["agentview_image"]
            elif "agentview" in obs:
                frame = obs["agentview"]
            else:
                # Try to find any camera image
                camera_keys = [k for k in obs.keys() if "image" in k.lower()]
                if camera_keys:
                    frame = obs[camera_keys[0]]
                    if i == 0:
                        print(f"Using camera key: {camera_keys[0]}")
                else:
                    raise KeyError(
                        f"No camera image found in observations. Available keys: {list(obs.keys())}"
                    )

            # May need to flip the image vertically
            flip_img = 1
            if robosuite.macros.IMAGE_CONVENTION == "opengl":
                flip_img = -1  # flip image to convert to opencv convention
            frame = frame[::flip_img, :, :]

            # Ensure frame is in the right format (H, W, 3) and uint8
            if frame.dtype != np.uint8:
                frame = (
                    (frame * 255).astype(np.uint8)
                    if frame.max() <= 1.0
                    else frame.astype(np.uint8)
                )

            # Handle different image formats
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # RGB image, good to go
                pass
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                # RGBA image, convert to RGB
                frame = frame[:, :, :3]
            else:
                raise ValueError(f"Unexpected image shape: {frame.shape}")

            frames.append(frame)

            if i == 0:
                print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

        except Exception as e:
            print(f"Error on reset {i}: {e}")
            # Use a black frame as fallback
            fallback_frame = np.zeros(
                (args.image_height, args.image_width, 3), dtype=np.uint8
            )
            frames.append(fallback_frame)

    # Save video
    print(f"Saving video to: {args.output_video}")
    os.makedirs(
        (
            os.path.dirname(args.output_video)
            if os.path.dirname(args.output_video)
            else "."
        ),
        exist_ok=True,
    )

    try:
        # Create video writer
        with imageio.get_writer(args.output_video, fps=args.fps, quality=8) as writer:
            for frame in frames:
                writer.append_data(frame)

        print(f"Video saved successfully!")
        print(f"Video details:")
        print(f"  - Frames: {len(frames)}")
        print(f"  - Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
        print(f"  - FPS: {args.fps}")
        print(f"  - Duration: {len(frames) / args.fps:.2f} seconds")

    except Exception as e:
        print(f"Error saving video: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Capture initial agentview frames from environment resets as video"
    )

    parser.add_argument(
        "--task_suite_name",
        type=str,
        default=None,
        help="Name of MimicLabs task suite (e.g., 'example_suite'). If None, task_name should be a robosuite environment name.",
    )

    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name of task within the suite (e.g., 'example_task') or robosuite environment name if no task_suite.",
    )

    parser.add_argument(
        "--num_resets",
        type=int,
        default=50,
        help="Number of environment resets to perform (number of frames to capture).",
    )

    parser.add_argument(
        "--output_video",
        type=str,
        default="initial_frames.mp4",
        help="Output video file path.",
    )

    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for output video."
    )

    parser.add_argument(
        "--robots", type=str, nargs="+", default=["Panda"], help="Robot types to use."
    )

    parser.add_argument(
        "--image_height", type=int, default=256, help="Height of captured images."
    )

    parser.add_argument(
        "--image_width", type=int, default=256, help="Width of captured images."
    )

    args = parser.parse_args()

    # Validate arguments
    if args.task_suite_name is not None:
        task_suite_path = os.path.join(
            mimiclabs.__path__[0], "mimiclabs", "task_suites", args.task_suite_name
        )
        if not os.path.exists(task_suite_path):
            raise ValueError(f"Task suite not found: {task_suite_path}")

    capture_reset_frames(args)


if __name__ == "__main__":
    main()
