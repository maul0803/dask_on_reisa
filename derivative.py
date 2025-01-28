import os
import numpy as np
from ray.util.dask import enable_dask_on_ray, disable_dask_on_ray
from reisa import Reisa
import dask.array as da

# The user can decide which task is executed on each level of the following tree
'''
   [p0 p1 p2 p3]-->[p0 p1 p2 p3]      # One task per process per iteration (we can get previous iterations data)
    \  /   \  /     \  /  \  /
     \/     \/       \/    \/
     iteration 0--> iteration 1 --> [...] --> result
    # One task per iteration (typically gathering the results of all the actors in that iteration)
    # We can get the result of the previous iterations
'''

# Get infiniband address
address = os.environ.get("RAY_ADDRESS").split(":")[0]
# Starting reisa (mandatory)
handler = Reisa("config.yml", address)
max = handler.iterations
    
# Process-level analytics code
def process_func(rank: int, i: int, queue):
    # "queue" will be the available data for "rank" process since the simulation has started (from iteration 0 to "i")
    # We must take into account that some data will be erased to free memory
    dask_queue = da.from_array(queue, chunks=5)
    gt = dask_queue[-5:]
    c0 = 2. / 3.
    result = da.average(c0 / 1 * (gt[3] - gt[1] - (gt[4] - gt[0]) / 8.))
    return result.compute()

# Iteration-level analytics code
def iter_func(i: int, current_results):
    dask_results = da.from_array(current_results, chunks=len(current_results))
    return dask_results.mean().compute()

def global_func(final_results):
    dask_results = da.from_array(final_results, chunks=len(final_results))
    return dask_results.mean().compute()


# Use our Dask config helper to set the scheduler to ray_dask_get globally,
# without having to specify it on each compute call.
enable_dask_on_ray()

# The iterations that will be executed (from 0 to end by default), in this case we will need 4 available timesteps
iterations = [i for i in range(4, max)]

# Launch the analytics (blocking operation), kept iters parameter means the number of iterations kept in memory before the current iteration
result = handler.get_result(process_func, iter_func, global_func=global_func, selected_iters=iterations, kept_iters=5, timeline=False)

# Not useful after the compute
disable_dask_on_ray()

# Write the results
with open("results.log", "a") as f:
    f.write("\nResult: "+str(result)+".\n")

handler.shutdown()