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
        'function': 'simulation_T_list'
    },
    'task_parameters': [],
    'requirements_slurm': {
        'time': '2-00:00:00', # d-hh:mm:ss
        'mem': '10G',
        'partition': 'cpu',
        'qos': 'normal',
        'nodes': 1,
        'cpus-per-task': 20,
    },
    'options': {}
}

# Choose which delta/T list set to use
delta_list_name = 'delta_list_3'
T_list_name = 'T_list_2'
delta_list = getattr(vps, delta_list_name)
T_list = getattr(vps, T_list_name)

# Choose multiple loop list names to sweep
loop_list_names = ['loop_halfsize11', 'loop_halfsize21', 'loop_halfsize31', 'loop_halfsize35']  # extend as needed
system_sizes = [11, 21, 31, 35]

N_shots = 15
N_cycles = 15
loop_type = 'general'
edge = True

for loop_list_name, system_size in zip(loop_list_names, system_sizes):
    loop_list = getattr(vps, loop_list_name)

    if len(loop_list_names) != len(system_sizes):
        raise ValueError("loop_list_names and system_sizes must have same length")


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
            kwargs = {
                'T_list': T_list,
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