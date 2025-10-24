import numpy as np

loop_0 = None
loop_1 = [9 for _ in range(11)]
loop_2 = [10 for _ in range(12)]
loop_3 = [11 for _ in range(13)]
loop_4 = [12 for _ in range(14)]
loop_5 = [13 for _ in range(15)]
loop_20 = [20 for _ in range(22)]
loop_30 = [30 for _ in range(32)]

loop_halfsize11 = [5 for _ in range(7)]
loop_halfsize21 = [10 for _ in range(12)]
loop_halfsize31 = [15 for _ in range(17)]
loop_halfsize35 = [17 for _ in range(19)]


delta_list_0 = [0,0.1,0.2,0.3]
delta_list_1 = [0,0.1,0.2,0.3,0.4,0.5]
delta_list_2 = [0,0.05,0.1,0.15,0.2,0.25,0.3]
delta_list_3 = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]

T_list_0 = np.linspace(1,0.1,10).tolist()
T_list_1 = np.linspace(1,0.1,19).tolist()
T_list_2 = np.linspace(1,0.1,25).tolist()
