"""Submit all (delta_list, T_list) combinations for multiple loop lists from variables_phasediagram_simulation.py."""
import copy
import numpy as np
import cluster_jobs
import variables_phasediagram_simulation as vps

config = {
    'jobname': 'PhaseDiagramMulti',
    'task': {
        'type': 'PythonFunctionCall',
        'module': 'PhaseDiagramCluster',
        'function': 'simulation'
    },
    'task_parameters': [],
    'requirements_slurm': {
        'time': '2-00:00:00', # d-hh:mm:ss
        'mem': '8G',
        'partition': 'cpu',
        'qos': 'normal',
        'nodes': 1,
        'cpus-per-task': 12,
    },
    'options': {}
}

# Choose which delta/T list set to use
delta_list_name = 'delta_list_3'
T_list_name = 'T_list_2'
delta_list = getattr(vps, delta_list_name)
T_list = getattr(vps, T_list_name)

# Choose multiple loop list names to sweep
loop_list_names = ['loop_0', 'loop_20', 'loop_30']  # extend as needed

N_shots = 15
system_size = 41
N_cycles = 15
loop_type = 'general'
edge = True

for loop_list_name in loop_list_names:
    loop_list = getattr(vps, loop_list_name)
    print("Using loop list:", loop_list_name)

    save_dir = (
        f"{T_list_name}"
        + f"_{delta_list_name}"
        + f"_size{system_size}"
        + f"_shots{N_shots}"
        + f"_cycles{N_cycles}"
        + ("_edge" if edge else "_noedge")
        + f"_{loop_type}"
        + f"_{loop_list_name}"
    )

    for delta in delta_list:
        print("computing delta: ", delta)
        for T in T_list:
            kwargs = {
                'T': T,
                'delta': delta,
                'N_cycles': N_cycles,
                'N_shots': N_shots,
                'system_size': system_size,
                'edge': edge,
                'save_dir': save_dir,
                'loop_type': loop_type,
                'loop_list_name': loop_list_name,
                'loop_list': loop_list,
            }
            config['task_parameters'].append(copy.deepcopy(kwargs))

cluster_jobs.SlurmJob(**config).submit()