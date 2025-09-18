"""Example how to create a `config` for a job array and submit it using cluster_jobs.py."""

import cluster_jobs
import copy
import numpy as np  # only needed if you use np below

config = {
    'jobname': 'PhaseDiagram',
    'task': {
        'type': 'PythonFunctionCall',
        'module': 'simulation',
        'function': 'run_simulation'
    },
    'task_parameters': [],  # list of dict containing the **kwargs given to the `function`
    'requirements_slurm': {  # passed on to SLURM
        'time': '1-23:00:00',  # d-hh:mm:ss
        'mem': '4G',
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

delta_list = [0,0.1,0.2,0.3]
T_list = np.linspace(1,0.1,10).tolist()
N_shots = 10
loop_list = [9 for _ in range(11)]
system_size = 31
N_cycles = 10
loop_type = 'general'
edge = True

save_dir = "pd" + f"_size{system_size}" + f"_Nshots{N_shots}" + f"_cycles{N_cycles}" + ("_edge" if edge else "_noedge") + f"_{loop_type}_loop" + (f"_{loop_list}" if loop_list is not None else "")



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
            'loop_list': loop_list
        }
        # ensure different output filename per simulation:
        # kwargs['output_filename'] = f'delta_{delta:.2f}_T_{T:.2f}.pkl'
        config['task_parameters'].append(copy.deepcopy(kwargs))

# cluster_jobs.TaskArray(**config).run_local(task_ids=[2, 3], parallel=2) # run selected tasks
# cluster_jobs.JobConfig(**config).submit()  # run all tasks locally by creating a bash job script
cluster_jobs.SlurmJob(**config).submit()  # submit to SLURM
# cluster_jobs.SGEJob(**config).submit()  # submit to SGE
