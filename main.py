import argparse
from data_process import DataProcess
from CDLib import CD_RUN
from utils import metric_dag_res, find_optimal_threshold_and_matrix_curve
import time


def main():
    parser = argparse.ArgumentParser(description='Main Function')
    # Dataset related parameters (including data simulation)
    parser.add_argument('--data_name', type=str, default='25V_474N_Microwave', choices=['data simulation', '18V_55N_Wireless', '24V_439N_Microwave', '25V_474N_Microwave'], help='Dataset name')
    parser.add_argument('--topo_status', default=True, help='Whether to use topology matrix (for real datasets); for 18V_55N_Wireless, set to False')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory path')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed (only effective for data simulation)')

    # Hyperparameters for topology simulation data
    # Defaults = {20, 40, 20000, 0.00003-0.00005, 0.02-0.03, 2}
    parser.add_argument('--n_nodes', type=int, default=20, help='Number of nodes in simulated data [5, 10, 15, 20, 25]')
    parser.add_argument('--n_edges', type=int, default=20, help='Number of edges in simulated data [5, 10, 15, 20, 25]')
    parser.add_argument('--topo_nodes', type=int, default=40, help='Number of nodes in simulated topology network [10, 20, 30, 40, 50]')
    parser.add_argument('--topo_edges', type=int, default=40, help='Number of edges in simulated topology network [10, 20, 30, 40, 50]')
    parser.add_argument('--mu_range_min', type=float, default=0.00003, help="mu: base event rate or spontaneous trigger rate")
    parser.add_argument('--mu_range_max', type=float, default=0.00005, help="mu: {1,2,3,4,5}*10e-5")
    parser.add_argument('--alpha_range_min', type=float, default=0.02, help="alpha: influence decay coefficient or propagation strength")
    parser.add_argument('--alpha_range_max', type=float, default=0.03, help="alpha: {1,2,3,4,5}*10e-2")
    parser.add_argument('--T', type=int, default=3600 * 24 * 2, help="Simulate 24-hour data (in seconds)")
    parser.add_argument('--sample_size', type=int, default=20000, help="Length of selected sequence [5000, 10000, 15000, 20000, 25000, 30000]")
    parser.add_argument('--max_hop', type=int, default=2, help="Maximum propagation hops: influence propagates at most 2 hops in the topology network")
    parser.add_argument('--delta', type=int, default=1, help="Window size [1,2,3,4,5]")
    parser.add_argument('--interval', type=int, default=1, help="Time interval [1,2,3,4,5]")

    # Non-topology simulation hyperparameters
    parser.add_argument('--method', type=str, default='linear', choices=['linear', 'nonlinear'], help='Simulation method (only effective for data simulation)')
    parser.add_argument('--sem_type', type=str, default='gauss', choices=['gauss', 'exp', 'gumbel'], help='Noise type (only effective for data simulation)')
    parser.add_argument('--weight_range_min', type=float, default=0.5, help='Minimum weight range (only effective for data simulation)')
    parser.add_argument('--weight_range_max', type=float, default=2.0, help='Maximum weight range (only effective for data simulation)')

    # Algorithm related parameters
    parser.add_argument('--algorithm', type=str, default='TACD', choices=['TACD'], help='Causal discovery algorithms')
    parser.add_argument('--max_iter', type=int, default=50, help='Number of training iterations')

    # Proposed hyper-params
    args = parser.parse_args()

    print("=" * 50)
    print("Parameter configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=" * 50)

    # Data processing
    topo_matrix, dag_matrix, X, alarm_data = DataProcess(args)

    # Run algorithm
    est_causal_matrix = CD_RUN(data_name=args.data_name, algorithm_name=args.algorithm, topo_matrix=topo_matrix, alarm_data=alarm_data, dag_matrix=dag_matrix)

    # Threshold curve
    best_threshold, est_causal_matrix = find_optimal_threshold_and_matrix_curve(est_causal_matrix, dag_matrix)
    TP, FN, TN, FP, PRE, REC, F1, g_score, shd = metric_dag_res(est_causal_matrix, dag_matrix)
    print(f"Train\t{PRE:.4f}\t{REC:.4f}\t{F1:.4f}\t{shd}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("consume_time", end_time - start_time)