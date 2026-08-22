from method.TACD import CDTES_Func

def CD_RUN(data_name, algorithm_name, topo_matrix, alarm_data, dag_matrix):
    print(f"Running {algorithm_name} ...")
    if algorithm_name == "TACD":
        est_causal_matrix = CDTES_Func(alarm_data, topo_matrix, dag_matrix, data_name)

    else:
        raise ValueError(f"No Support: {algorithm_name}")
    return est_causal_matrix