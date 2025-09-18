"""Submit a single (delta_list, T_list, loop_list) configuration from variables_phasediagram_simulation.py."""

import cluster_jobs
import copy
import numpy as np  # only needed if you use np below
import variables_phasediagram_simulation as vps

config = {
    'jobname': 'PhaseDiagram',
    'task': {
        'type': 'PythonFunctionCall',
        'module': 'PhaseDiagramCluster',
        'function': 'simulation'
    },
    'task_parameters': [],  # list of dict containing the **kwargs given to the `function`
    'requirements_slurm': {  # passed on to SLURM
        'time': '1-23:00:00',  # d-hh:mm:ss
        'mem': '3G',
        'partition': 'cpu',
        'qos': 'normal',
        'nodes': 1,  # number of nodes
        'cpus-per-task': 16,  # number of cpus per task
    },
    #  'requirements_sge': {  # for SGE
    #      'l': 'h_cpu=0:30:00,h_rss=4G',
    #      'q': 'queue',
    #      'pe smp': '4',
    #      # 'M': "no@example.com"
    #  },
    'options': {
        # you can add extra variables for the script_template in cluster_templates/* here
    }
}

# Select which lists to use by name from variables_phasediagram_simulation.py
delta_list_name = 'delta_list_2'
T_list_name = 'T_list_1'
# loop_list_name = 'loop_1'

delta_list = getattr(vps, delta_list_name)
T_list = getattr(vps, T_list_name)
# loop_list = getattr(vps, loop_list_name)

N_shots = 15
system_size = 31
N_cycles = 10
loop_type = 'parallelogram'
edge = True

save_dir = (
    "pd"
    + f"_size{system_size}"
    + f"_Nshots{N_shots}"
    + f"_cycles{N_cycles}"
    + ("_edge" if edge else "_noedge")
    + f"_{loop_type}"
    + f"_{loop_list_name}"
)


for delta in delta_list:
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
            'loop_list_name': loop_list_name,  # pass the name (for logging/paths)
            'loop_list': loop_list
        }
        config['task_parameters'].append(copy.deepcopy(kwargs))

# cluster_jobs.TaskArray(**config).run_local(task_ids=[2, 3], parallel=2) # run selected tasks
# cluster_jobs.JobConfig(**config).submit()  # run all tasks locally by creating a bash job script
cluster_jobs.SlurmJob(**config).submit()  # submit to SLURM
# cluster_jobs.SGEJob(**config).submit()  # submit to SGE
