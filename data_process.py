import numpy as np
import pandas as pd
import random
from castle.datasets import DAG, Topology, THPSimulation
from data_mapto_es import downsample_by_time_interval

def DataProcess(args):
    data_name, topo_status = args.data_name, args.topo_status
    if data_name == '24V_439N_Microwave' or data_name == '25V_474N_Microwave':
        alarm_path = "./dataset/"+data_name+"/alarm.csv"
        topo_path = "./dataset/"+data_name+"/topology.npy"
        dag_path = "./dataset/"+data_name+"/true_graph.npy"

        alarm_data = pd.read_csv(alarm_path, encoding='utf')
        if topo_status:
            topo_matrix = np.load(topo_path)
        else:
            topo_matrix = None
        dag_matrix = np.load(dag_path)
        X = alarm_data.iloc[:, 0:3]
        X.columns = ['event', 'node', 'timestamp']
        X = X.reindex(columns=['event', 'timestamp', 'node'])

    elif data_name == '18V_55N_Wireless':
        alarm_path = "./dataset/"+data_name+"/Alarm.csv"
        dag_path = "./dataset/"+data_name+"/DAG.npy"
        alarm_data = pd.read_csv(alarm_path, encoding='utf')
        if topo_status:
            n_nodes = 55
            topo_matrix = np.ones((n_nodes, n_nodes), dtype=int)
            np.fill_diagonal(topo_matrix, 0)
        else:
            topo_matrix = None
        dag_matrix = np.load(dag_path)
        X = alarm_data.iloc[:, 0:3]
        X.columns = ['event', 'node', 'timestamp']
        X = X.reindex(columns=['event', 'timestamp', 'node'])

    else:
        print("No input dataset!")

    return topo_matrix, dag_matrix, X, alarm_data
