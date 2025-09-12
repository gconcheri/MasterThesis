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
        'time': '0-00:30:00',  # d-hh:mm:ss
        'mem': '30G',
        'partition': 'cpu',
        'qos': 'normal',
        'nodes': 1,  # number of nodes
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

for delta in delta_list:
    for T in T_list:
        kwargs = {
            'delta': delta,
            'T': T,
            'N_cycles': 10,
            'N_shots': N_shots,
            'system_size': 31,
            'edge': True
        }
        # ensure different output filename per simulation:
        # kwargs['output_filename'] = f'delta_{delta:.2f}_T_{T:.2f}.pkl'
        config['task_parameters'].append(copy.deepcopy(kwargs))

# cluster_jobs.TaskArray(**config).run_local(task_ids=[2, 3], parallel=2) # run selected tasks
# cluster_jobs.JobConfig(**config).submit()  # run all tasks locally by creating a bash job script
cluster_jobs.SlurmJob(**config).submit()  # submit to SLURM
# cluster_jobs.SGEJob(**config).submit()  # submit to SGE
