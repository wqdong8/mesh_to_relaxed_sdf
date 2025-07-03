import multiprocessing
import subprocess
import os
import time
import argparse
import json
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
Example command:

python distributed_process.py --input_models_info path/to/your/data_info.json --save_dir RSDF_datasets --num_gpus 8 --workers_per_gpu 4 --gpu_list 0 1 2 3 4 5 6 7 --save_watertight_mesh --end_i -1 --remesh_resolution 512

This script processes 3D models to generate watertight meshes and sample points for SDF computation.
"""

parser = argparse.ArgumentParser(description='distributed sampling')

parser.add_argument('--input_models_info', type=str,
                    help='Path to a json file containing a list of 3D object files.')
parser.add_argument('--save_dir', type=str,
                    help='Path to save result.')
parser.add_argument('--num_sample', type=int, default=200000,
                    help='number of sample point.')
parser.add_argument('--num_gpus', type=int, default=-1,
                    help='number of gpus to use. -1 means all available gpus.')
parser.add_argument('--gpu_list',nargs='+', type=int, 
                    help='the avalaible gpus')
parser.add_argument('--workers_per_gpu', type=int, default=1,
                    help='number of workers per gpu.')
parser.add_argument('--start_i', type=int, default=0,
                    help='number of start index.')
parser.add_argument('--end_i', type=int, default=-1,
                    help='number of end index.')
parser.add_argument('--remesh_resolution', type=int, default=512,
                    help='Resolution for watertight mesh generation.')
parser.add_argument('--save_watertight_mesh', action='store_true',
                    help='Whether to save watertight mesh.')
parser.add_argument('--num_split', type=int, default=1,
                    help='number of split for sampling.')
args = parser.parse_args()

def check_task_finish(model_id, strict=True):
    """Check if processing for a model is already completed."""
    if strict:
        # Check if all expected output files exist
        num_splits = args.num_split
        file_name = [f"sample_{i}.npz" for i in range(num_splits)]
        sample_path = os.path.join(args.save_dir, "samples", model_id)
        for file in file_name:
            if not os.path.exists(os.path.join(sample_path, file)):
                return False
        return True
    else:
        # Check if the number of files matches expected split count
        sample_path = os.path.join(args.save_dir, "samples", model_id)
        if len(os.listdir(sample_path)) == args.num_split:
            return True
        else:
            return False

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
) -> None:
    """Worker process function that processes mesh files on a specific GPU."""
    while True:
        item = queue.get()
        if item is None:
            break
        
        # Unpack the item
        obj_path, model_id = item
        logger.info(f"Processing: {item} on gpu {gpu}")
        # Skip already processed models
        if check_task_finish(model_id):
            queue.task_done()
            logger.info(f'======== {model_id} already processed ========')
            continue
        
        try:
            # Create directories for outputs
            watertight_dir = os.path.join(args.save_dir, "watertight")
            samples_dir = os.path.join(args.save_dir, "samples")
            
            os.makedirs(watertight_dir, exist_ok=True)
            os.makedirs(samples_dir, exist_ok=True)
            
            # Step 1: Generate watertight mesh from input model
            logger.info(f"Creating watertight mesh for {model_id}")
            watertight_cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} "
                f"python3 to_watertight_mesh.py "
                f"input.mesh_path={obj_path} "
                f"input.mesh_id={model_id} "
                f"output.save_dir={watertight_dir} "
                f"processing.resolution={args.remesh_resolution} "
                f"processing.scale=1.0 "
            )
            # logger.info(watertight_cmd)
            watertight_result = subprocess.run(watertight_cmd, shell=True, capture_output=True, text=True)
            if watertight_result.returncode != 0:
                logger.error(f"Error creating watertight mesh for {model_id}: {watertight_result.stderr}")
                queue.task_done()
                continue
            
            # Step 2: Sample points from the watertight mesh
            watertight_mesh_path = os.path.join(watertight_dir, f'{model_id}.obj')
            if not os.path.exists(watertight_mesh_path):
                logger.error(f"Watertight mesh not found for {model_id}")
                queue.task_done()
                continue
                
            logger.info(f"Sampling from watertight mesh for {model_id}")
            sample_cmd = (
                f"python3 mesh_sample.py "
                f"input.mesh_path={watertight_mesh_path} "
                f"input.mesh_id={model_id} "
                f"sampling.point_number={args.num_sample} "
                f"output.save_dir={os.path.join(samples_dir, model_id)} "
                f"output.num_split={args.num_split} "
            )
            # logger.info(sample_cmd)
            sample_result = subprocess.run(sample_cmd, shell=True, capture_output=True, text=True)
            if sample_result.returncode != 0:
                logger.error(f"Error sampling from watertight mesh for {model_id}: {sample_result.stderr}")
                queue.task_done()
                continue
            
            # Step 3: Clean up watertight mesh if not needed
            if not args.save_watertight_mesh:
                if os.path.exists(watertight_mesh_path):
                    os.system(f"rm {watertight_mesh_path}")
                    logger.info(f"Deleted watertight mesh directory for {watertight_mesh_path}")
            
            # Update counter for successful processing
            with count.get_lock():
                count.value += 1
                
        except Exception as e:
            logger.error(f"Error processing {model_id}: {e}")
        finally:
            queue.task_done()

if __name__ == "__main__":
    # Initialize multiprocessing queue and counter
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            process = multiprocessing.Process(
                target=worker, args=(queue, count, args.gpu_list[gpu_i])
            )
            process.daemon = True
            process.start()
    
    # Load model list from JSON file
    with open(args.input_models_info, 'r') as f:
        model_items = json.load(f)

    logger.info(f"Total found {len(model_items)} objects.")
    
    # Set end index if not specified
    if args.end_i == -1 or args.end_i > len(model_items):
        args.end_i = len(model_items)

    # Select models based on start and end indices
    model_items = model_items[args.start_i:args.end_i]
    logger.info(f"Selected {len(model_items)} objects to process.")
    
    # Add items to the queue for processing
    for obj_path, model_id in model_items:
        queue.put((obj_path, model_id))
    
    start_time = time.time()

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinel values to terminate worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)

    end_time = time.time()
    logger.info("All Processing Finished in %s seconds", end_time - start_time)
    logger.info("Total %s objects successfully processed.", count.value)