job_setup_time = 50  # seconds
load_split_time_per_task = 5  # seconds
map_time_per_task = 15  # seconds
copy_merge_time = 150  # seconds
reduce_time_per_task = 50  # seconds
write_time_per_task = 10  # seconds

# Number of tasks and workers
num_map_tasks = 1000
num_workers = 200

# Calculate the time taken for the load split and map stages
# Assuming equal distribution of map tasks across workers
tasks_per_worker = num_map_tasks / num_workers
time_per_worker_for_map = tasks_per_worker * (load_split_time_per_task + map_time_per_task)

# Total time for the job
# The longest stage determines the total time. In this case, it's either the map stage or the copy/merge stage.
total_time = max(job_setup_time, time_per_worker_for_map, copy_merge_time) + reduce_time_per_task + write_time_per_task
total_time

print("Total time taken for the job: {} seconds".format(total_time)