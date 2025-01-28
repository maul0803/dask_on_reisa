import os
import dask.array as da
import ray
from ray.util.dask import enable_dask_on_ray, disable_dask_on_ray
from reisa import Reisa
import numpy as np


# Initialization of REISA
address = os.environ.get("RAY_ADDRESS").split(":")[0]
handler = Reisa("config.yml", address)#Ray.init
max = handler.iterations

# Process-level analytics code (Dask)
def process_func(rank: int, i: int, queue):
    data = ray.get(queue[i]) if isinstance(queue[i], ray.ObjectRef) else queue[i]
    # Use of a dask array (instead of a numpy array)
    gt = da.from_array(data, chunks=5)
    return gt.sum().compute()  # Execute the computation

# Iteration-level analytics code (Dask)
def iter_func(i: int, current_results):
    data = ray.get(current_results)
    results_dask = da.from_array(data, chunks=len(current_results))
    return results_dask.sum().compute()  # Perform the computation


# Use our Dask config helper to set the scheduler to ray_dask_get globally,
# without having to specify it on each compute call.
enable_dask_on_ray()

# Iterations to be executed
iterations = [i for i in range(0, max)]

# Launch the analytics with Dask
result = handler.get_result(process_func, iter_func, selected_iters=iterations, kept_iters=max, timeline=False)

# Not useful after the compute
disable_dask_on_ray()

# Save the results
with open("results.log", "a") as f:
    f.write("\nResults per iteration: " + str(result) + ".\n")

handler.shutdown()
