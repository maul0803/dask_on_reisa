import time
import numpy as np
from reisa import Reisa  # Mandatory import
import os
from ray.util.dask import ray_dask_get, enable_dask_on_ray, disable_dask_on_ray
import dask.array as da
import ray
# The user can decide which task is executed on each level of the following tree
"""
   [p0 p1 p2 p3]-->[p0 p1 p2 p3]      # One task per process per iteration (we can get previous iterations' data)
    \  /   \  /     \  /  \  /
     \/     \/       \/    \/
     iteration 0--> iteration 1 --> [...] --> result
    # One task per iteration (typically gathering the results of all actors in that iteration)
    # We can get the result of previous iterations.
"""

# Get infiniband address
address = os.environ.get("RAY_ADDRESS").split(":")[0]
# Starting reisa (mandatory)
handler = Reisa("config.yml", address)
max_iterations = handler.iterations

# Process-level analytics code
def process_func(rank: int, i: int, queue):
    """
    Process-level function executed for each process in an iteration.

    :param rank: The rank (ID) of the process.
    :param i: The current iteration index.
    :param queue: Data queue containing simulation values for the iteration.
    :return: Dask array with the sum of the queue's data for the current iteration.
    """
    gt = ray.get(queue[-5:])
    c0 = 2. / 3.
    result = c0 * (gt[3] - gt[1] - (gt[4] - gt[0]) / 8.)
    result_dask = da.mean(result)
    return result_dask


# Iteration-level analytics code
def iter_func(i: int, current_results):
    """
    Iteration-level function that aggregates results from all processes.

    :param i: The current iteration index.
    :param current_results: List of results from all processes in the iteration.
    :return: Sum of all process results for the iteration.
    """
    current_results = current_results.mean().compute(scheduler=ray_dask_get)
    return current_results


def global_func(final_results):
    """
    Global-level function that computes the overall average of all iteration results.

    :param final_results: A list of results from all iterations.
    :return: The average value of all iteration results.
    """
    return np.average(final_results)


# Define the range of iterations to be executed (default is from 0 to max)
iterations = list(range(4, max_iterations))

# Launch the analytics (blocking operation), kept iters paramerter means the number of iterations kept in memory before the current iteration
result = handler.get_result(process_func, iter_func, global_func=global_func, selected_iters=iterations, kept_iters=5, timeline=False)

# Write the results to a log file
with open("results.log", "a") as f:
    f.write("\nResult: "+str(result)+".\n")

# Shut down the handler and free resources
handler.shutdown()