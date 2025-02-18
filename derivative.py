import time
import numpy as np
from reisa import Reisa  # Mandatory import
import os
from ray.util.dask import ray_dask_get, enable_dask_on_ray, disable_dask_on_ray
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
    gt = queue[-5:]
    c0 = 2. / 3.
    result = c0 * (gt[3] - gt[1] - (gt[4] - gt[0]) / 8.)
    result_dask = da.mean(result)
    return result_dask


# Iteration-level analytics code
def iter_func(i: int, current_results):
    current_results = current_results.mean().compute(scheduler=ray_dask_get)
    return current_results


def global_func(final_results):
    return np.average(final_results)


# The iterations that will be executed (from 0 to end by default), in this case we will need 4 available timesteps
iterations = [i for i in range(4, max)]

# Launch the analytics (blocking operation), kept iters paramerter means the number of iterations kept in memory before the current iteration
result = handler.get_result(process_func, iter_func, global_func=global_func, selected_iters=iterations, kept_iters=5,
                            timeline=False)

# Write the results
with open("results.log", "a") as f:
    f.write("\nResult: " + str(result) + ".\n")

handler.shutdown()