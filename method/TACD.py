import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import deque
import random
from utils import metric_dag_res, find_optimal_threshold_and_matrix_curve
from config_hparam import config_sim, config_18V, config_24V, config_25V

DATASET_CONFIG_MAP = {
    'data simulation': config_sim,
    '18V_55N_Wireless': config_18V,
    '24V_439N_Microwave': config_24V,
    '25V_474N_Microwave': config_25V
}

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dag_ness(A):
    n = A.shape[0]
    A_hadamard = A * A
    exp_A = torch.linalg.matrix_exp(A_hadamard)
    h = torch.trace(exp_A) - n
    return h


def find_k_hop_neighbors(adj_matr, k):
    adj_matr = np.array(adj_matr)
    n = len(adj_matr)
    neighbors_dict = {i: [i] for i in range(n)}
    for node in range(n):
        visited = [False] * n
        visited[node] = True
        queue = deque([(node, 0)])
        while queue:
            curr_node, curr_dist = queue.popleft()
            if curr_dist > k:
                break
            for neighbor in range(n):
                if adj_matr[curr_node, neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    neighbors_dict[node].append(neighbor)
                    queue.append((neighbor, curr_dist + 1))
    return neighbors_dict


def build_k_hop_matrix(topo_matrix, k):
    neighbors_dict = find_k_hop_neighbors(topo_matrix, k)
    n = len(topo_matrix)
    B = np.zeros((n, n))
    for i, neigh_list in neighbors_dict.items():
        B[i, neigh_list] = 1
    return torch.tensor(B, dtype=torch.float)


# -------------------- Gumbel-Sigmoid Function --------------------
def gumbel_sigmoid(logits, tau=1.0, hard=False):
    ''' Using the Gumbel-Sigmoid for Continuously Differentiable Sampling from a Binomial Distribution '''
    gumbel_noise_1 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    gumbel_noise_0 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y_soft = torch.sigmoid((logits + gumbel_noise_1 - gumbel_noise_0) / tau)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = y_hard - y_soft.detach() + y_soft  # Straight-through estimator
        return y
    return y_soft


def prepare_event_pairs(alarm_data, topo_matrix, k_hop=2):
    num_devices = topo_matrix.shape[0]
    nei_dict = find_k_hop_neighbors(topo_matrix, k_hop)
    device_groups = alarm_data.groupby('device_id')

    all_sequences = []
    max_end = 0
    for dev in range(num_devices):
        if dev in device_groups.groups:
            df = device_groups.get_group(dev).sort_values('start_timestamp')
            types = df['alarm_id'].values.astype(np.int64)
            starts = df['start_timestamp'].values.astype(np.float32)
            ends = df['end_timestamp'].values.astype(np.float32)
            all_sequences.append((dev, types, starts, ends))
            max_end = max(max_end, ends.max())
        else:
            all_sequences.append((dev, np.array([]), np.array([]), np.array([])))

    total_T = max_end
    event_pairs = []

    for target_dev, types, starts, ends in all_sequences:
        if len(types) == 0:
            continue
        neighbor_devices = nei_dict[target_dev]
        hist_events = []
        for nd in neighbor_devices:
            dev_data = all_sequences[nd]
            if len(dev_data[1]) == 0:
                continue
            h_dev, h_types, h_starts, h_ends = dev_data
            for j in range(len(h_types)):
                hist_events.append((h_types[j], nd, h_starts[j], h_ends[j]))

        hist_events.sort(key=lambda x: x[2])
        for i in range(len(types)):
            t_target = starts[i]
            target_type = types[i]
            hist_list = []
            for (h_type, h_dev, h_start, h_end) in hist_events:
                if h_start < t_target and h_end > t_target:
                    hist_list.append((h_type, h_dev, h_start, h_end))
            event_pairs.append((target_type, target_dev, t_target, hist_list))

    return event_pairs, total_T



class EventPairDataset(torch.utils.data.Dataset):
    def __init__(self, event_pairs, num_types):
        self.pairs = event_pairs
        self.num_types = num_types

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target_type, target_dev, target_time, hist_list = self.pairs[idx]
        if len(hist_list) > 0:
            hist_types = torch.tensor([h[0] for h in hist_list], dtype=torch.long)
            hist_devs = torch.tensor([h[1] for h in hist_list], dtype=torch.long)
            hist_starts = torch.tensor([h[2] for h in hist_list], dtype=torch.float)
            hist_ends = torch.tensor([h[3] for h in hist_list], dtype=torch.float)
        else:
            hist_types = torch.empty(0, dtype=torch.long)
            hist_devs = torch.empty(0, dtype=torch.long)
            hist_starts = torch.empty(0, dtype=torch.float)
            hist_ends = torch.empty(0, dtype=torch.float)
        return {
            'target_type': torch.tensor(target_type, dtype=torch.long),
            'target_dev': torch.tensor(target_dev, dtype=torch.long),
            'target_time': torch.tensor(target_time, dtype=torch.float),
            'hist_types': hist_types,
            'hist_devs': hist_devs,
            'hist_starts': hist_starts,
            'hist_ends': hist_ends
        }


def collate_fn(batch):
    target_type = torch.stack([b['target_type'] for b in batch])
    target_dev = torch.stack([b['target_dev'] for b in batch])
    target_time = torch.stack([b['target_time'] for b in batch])
    max_len = max(len(b['hist_types']) for b in batch)
    batch_size = len(batch)
    hist_types = torch.zeros(batch_size, max_len, dtype=torch.long)
    hist_devs = torch.zeros(batch_size, max_len, dtype=torch.long)
    hist_starts = torch.zeros(batch_size, max_len, dtype=torch.float)
    hist_ends = torch.zeros(batch_size, max_len, dtype=torch.float)
    hist_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        l = len(b['hist_types'])
        if l > 0:
            hist_types[i, :l] = b['hist_types']
            hist_devs[i, :l] = b['hist_devs']
            hist_starts[i, :l] = b['hist_starts']
            hist_ends[i, :l] = b['hist_ends']
            hist_mask[i, :l] = True
    return {
        'target_type': target_type, 'target_dev': target_dev, 'target_time': target_time,
        'hist_types': hist_types, 'hist_devs': hist_devs, 'hist_starts': hist_starts,
        'hist_ends': hist_ends, 'hist_mask': hist_mask
    }


def group_event_pairs(event_pairs):
    grouped = {}
    for pair in event_pairs:
        target_type, target_dev, target_time, hist_list = pair
        if len(hist_list) > 0:
            hist_types = torch.tensor([h[0] for h in hist_list], dtype=torch.long)
            hist_devs = torch.tensor([h[1] for h in hist_list], dtype=torch.long)
            hist_starts = torch.tensor([h[2] for h in hist_list], dtype=torch.float)
            hist_ends = torch.tensor([h[3] for h in hist_list], dtype=torch.float)
            hist_mask = torch.ones(len(hist_list), dtype=torch.bool)
        else:
            hist_types = torch.empty(0, dtype=torch.long)
            hist_devs = torch.empty(0, dtype=torch.long)
            hist_starts = torch.empty(0, dtype=torch.float)
            hist_ends = torch.empty(0, dtype=torch.float)
            hist_mask = torch.empty(0, dtype=torch.bool)

        sample = {
            'target_dev': target_dev, 'target_time': target_time,
            'hist_types': hist_types, 'hist_devs': hist_devs,
            'hist_starts': hist_starts, 'hist_ends': hist_ends, 'hist_mask': hist_mask
        }
        grouped.setdefault(target_type, []).append(sample)
    return grouped


# -------------------- attention moudle --------------------
class UnifiedAttention(nn.Module):
    def __init__(self, num_types, num_devices, type_emb_dim=16, dev_emb_dim=8, use_time_decay=False):
        super().__init__()
        self.use_time_decay = use_time_decay
        self.type_emb = nn.Embedding(num_types, type_emb_dim)
        self.dev_emb = nn.Embedding(num_devices, dev_emb_dim)
        total_dim = type_emb_dim + dev_emb_dim
        self.W = nn.Parameter(torch.randn(total_dim, total_dim) * 0.01)

        if self.use_time_decay:
            self.base_decay = nn.Embedding(num_types, 1)
            nn.init.constant_(self.base_decay.weight, 0.05)

    def forward(self, hist_type, hist_dev, target_type, target_dev, time_diff=None):
        h_type_emb = self.type_emb(hist_type)
        h_dev_emb = self.dev_emb(hist_dev)

        if target_type.dim() == 2:
            t_type_emb = self.type_emb(target_type)
        else:
            t_type_emb = self.type_emb(target_type).unsqueeze(1).expand(-1, hist_type.size(1), -1)

        if target_dev.dim() == 2:
            t_dev_emb = self.dev_emb(target_dev)
        else:
            t_dev_emb = self.dev_emb(target_dev).unsqueeze(1).expand(-1, hist_type.size(1), -1)

        h_concat = torch.cat([h_type_emb, h_dev_emb], dim=-1)
        t_concat = torch.cat([t_type_emb, t_dev_emb], dim=-1)
        score = torch.einsum('...i,ij,...j->...', h_concat, self.W, t_concat)

        attention_score = F.softplus(score)

        # time decay
        if self.use_time_decay and time_diff is not None:
            if target_type.dim() == 2:
                decay_rates = F.softplus(self.base_decay(target_type).squeeze(-1))
            else:
                decay_rates = F.softplus(self.base_decay(target_type))
                decay_rates = decay_rates.expand(-1, hist_type.size(1))
            time_decay = torch.exp(-decay_rates * time_diff)
            attention_score = attention_score * time_decay

        return attention_score

class TACDGC(nn.Module):
    def __init__(self, num_event_types, num_devices, topo_matrix, dag_matrix, config):
        super().__init__()
        self.num_event_types = num_event_types
        self.num_devices = num_devices
        self.dag_matrix = dag_matrix
        self.config = config

        self.lambda1 = config.get('lambda1', 5e-3)
        self.lambda2 = config.get('lambda2', 1e-3)
        self.register_buffer('B', build_k_hop_matrix(topo_matrix, config.get('k_hop', 2)))

        if self.config.get('use_gumbel', False):
            self.A_logits = nn.Parameter(torch.zeros(num_event_types, num_event_types))
            self.A_weight = nn.Parameter(torch.randn(num_event_types, num_event_types) * 0.01)
            self.gumbel_tau = 1.0
            self.hard_gumbel = False
        else:
            self.A = nn.Parameter(torch.randn(num_event_types, num_event_types) * 0.01)

        if self.config.get('mu_type', 'shared') == 'device_specific':
            self.mu = nn.Parameter(torch.randn(num_devices, num_event_types) * 0.01)
        else:
            self.mu = nn.Parameter(torch.randn(num_event_types) * 0.01)

        self.attention = UnifiedAttention(
            num_types=num_event_types,
            num_devices=num_devices,
            type_emb_dim=config.get('type_emb_dim', 16),
            dev_emb_dim=config.get('dev_emb_dim', 8),
            use_time_decay=config.get('use_time_decay', False)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.get('lr', 1e-3), weight_decay=1e-5)

        # Soft Constraint Mask Properties
        self.use_order_mask = False
        self.hard_mask = None
        self.tau = 1.0
        self.order = None

    def compute_order_from_A(self, A_matrix):
        out_degree = A_matrix.sum(dim=1)
        in_degree = A_matrix.sum(dim=0)
        net = out_degree - in_degree
        order = torch.argsort(net, descending=True).cpu().numpy().tolist()
        return order

    def generate_hard_mask(self, order):
        mask = torch.zeros(self.num_event_types, self.num_event_types, device=device)
        for i, src in enumerate(order):
            for j, tgt in enumerate(order):
                if i < j:
                    mask[src, tgt] = 1.0
        return mask

    def apply_soft_mask(self, A_matrix):
        if self.hard_mask is None:
            return A_matrix
        soft_mask = self.hard_mask + self.tau * (1 - self.hard_mask)
        return A_matrix * soft_mask

    def get_effective_A(self, is_training=False):
        if self.config.get('use_gumbel', False):
            if is_training:
                A_prob = gumbel_sigmoid(self.A_logits, tau=self.gumbel_tau, hard=self.hard_gumbel)
            else:
                A_prob = torch.sigmoid(self.A_logits)
            A_eff = A_prob * F.softplus(self.A_weight)
        else:
            A_eff = F.softplus(self.A)

        if self.use_order_mask:
            A_eff = self.apply_soft_mask(A_eff)

        A_eff = A_eff * (1 - torch.eye(self.num_event_types, device=device))
        return A_eff

    def forward(self, batch, total_T):
        target_type = batch['target_type']
        target_dev = batch['target_dev']
        target_time = batch['target_time']
        hist_types = batch['hist_types']
        hist_devs = batch['hist_devs']
        hist_starts = batch['hist_starts']
        hist_ends = batch['hist_ends']
        hist_mask = batch['hist_mask']

        B, L = hist_types.shape
        topo_reachable = self.B[hist_devs, target_dev.unsqueeze(1)]
        time_diff = torch.relu(target_time.unsqueeze(1) - hist_starts)

        target_type_expanded = target_type.unsqueeze(1).expand(B, L)
        target_dev_expanded = target_dev.unsqueeze(1).expand(B, L)

        raw_excitation = self.attention(hist_types, hist_devs, target_type_expanded, target_dev_expanded, time_diff)

        A_eff = self.get_effective_A(is_training=self.training)
        causal_strength = A_eff[hist_types, target_type_expanded]

        contribution = raw_excitation * causal_strength * topo_reachable * hist_mask.float()
        excitation_sum = contribution.sum(dim=1)

        # Adaptive: Device-specific or shared globally
        mu_vals = F.softplus(self.mu)
        if self.config.get('mu_type', 'shared') == 'device_specific':
            baseline = mu_vals[target_dev, target_type]
        else:
            baseline = mu_vals[target_type]

        lambda_t = baseline + excitation_sum
        eps = 1e-8
        log_likelihood = torch.log(lambda_t + eps)

        baseline_integral = total_T * mu_vals.sum()
        hist_durations = hist_ends - hist_starts
        hist_integral = (contribution * hist_durations).sum(dim=1)

        nll = -(log_likelihood - (baseline_integral + hist_integral)).mean()
        l1_loss = torch.norm(A_eff, p=1)
        dag_loss = dag_ness(A_eff)

        total_loss = nll + self.lambda1 * l1_loss + self.lambda2 * dag_loss
        return total_loss, A_eff

    def _compute_node_loss(self, target_node, samples, A_matrix):
        total_loss = 0.0
        with torch.no_grad():
            for s in samples:
                target_type = torch.tensor([target_node], device=device)
                target_dev = torch.tensor([s['target_dev']], device=device)
                target_time = torch.tensor([s['target_time']], device=device)
                hist_types = s['hist_types'].unsqueeze(0).to(device)
                hist_devs = s['hist_devs'].unsqueeze(0).to(device)
                hist_starts = s['hist_starts'].unsqueeze(0).to(device)
                hist_ends = s['hist_ends'].unsqueeze(0).to(device)
                hist_mask = s['hist_mask'].unsqueeze(0).to(device)

                B, L = hist_types.shape
                topo_reachable = self.B[hist_devs, target_dev.unsqueeze(1)]
                time_diff = torch.relu(target_time.unsqueeze(1) - hist_starts)

                target_type_expanded = target_type.unsqueeze(1).expand(B, L)
                target_dev_expanded = target_dev.unsqueeze(1).expand(B, L)

                raw_excitation = self.attention(hist_types, hist_devs, target_type_expanded, target_dev_expanded,
                                                time_diff)

                A_nonneg = A_matrix * (1 - torch.eye(self.num_event_types, device=device))
                causal_strength = A_nonneg[hist_types, target_type_expanded]
                contribution = raw_excitation * causal_strength * topo_reachable * hist_mask.float()
                excitation_sum = contribution.sum(dim=1)

                mu_vals = F.softplus(self.mu)
                if self.config.get('mu_type', 'shared') == 'device_specific':
                    baseline = mu_vals[target_dev, target_type]
                else:
                    baseline = mu_vals[target_type]

                lambda_t = baseline + excitation_sum
                eps = 1e-8
                loss = -torch.log(lambda_t + eps).item()
                total_loss += loss
        return total_loss / len(samples) if samples else 0.0

    def evaluate(self, data_loader, total_T):
        self.eval()
        total_loss = 0
        valid_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, _ = self.forward(batch, total_T)
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    valid_batches += 1
        return total_loss / max(valid_batches, 1)

    def _get_scheduled_lr(self, epoch, default_lr):
        schedule = self.config.get('lr_schedule', None)
        if not schedule: return default_lr
        sorted_epochs = sorted(schedule.keys())
        lr = default_lr
        for ep in sorted_epochs:
            if epoch >= ep:
                lr = schedule[ep]
        return lr

    def train_model(self, train_loader, val_loader, val_grouped, total_T):
        phase1_epochs = self.config.get('phase1_epochs', 50)
        phase2_epochs = self.config.get('phase2_epochs', 5)
        patience = self.config.get('patience', 10)
        default_lr = self.config.get('lr', 1e-3)

        phase1_best_metric = float('-inf') if self.dag_matrix is not None else float('inf')
        global_best_metric = phase1_best_metric

        phase1_best_model_state = None
        global_best_model_state = None
        epochs_no_improve = 0

        tau_init = 1.0
        tau_min = 0.05

        for epoch in range(phase1_epochs):
            new_lr = self._get_scheduled_lr(epoch, default_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            if self.config.get('use_gumbel', False):
                self.gumbel_tau = tau_init * (tau_min / tau_init) ** (epoch / max(phase1_epochs - 1, 1))

            self.train()
            total_loss = 0
            valid_batches = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                loss, _ = self.forward(batch, total_T)
                if torch.isnan(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                valid_batches += 1

            avg_train_loss = total_loss / max(valid_batches, 1)

            if self.dag_matrix is not None:
                exp_A = self.get_effective_A(is_training=False).detach().cpu().numpy()
                _, est_causal_matrix = find_optimal_threshold_and_matrix_curve(exp_A, self.dag_matrix)
                TP, FN, TN, FP, PRE, REC, F1, g_score, shd = metric_dag_res(est_causal_matrix, self.dag_matrix)
                current_metric = F1

                if current_metric > phase1_best_metric:
                    phase1_best_metric = current_metric
                    phase1_best_model_state = {k: v.clone() for k, v in self.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if current_metric > global_best_metric:
                    global_best_metric = current_metric
                    global_best_model_state = {k: v.clone() for k, v in self.state_dict().items()}

            else:
                val_loss = self.evaluate(val_loader, total_T)
                current_metric = val_loss
                print(f"Epoch {epoch + 1}/{phase1_epochs}, Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

                if current_metric < phase1_best_metric:
                    phase1_best_metric = current_metric
                    phase1_best_model_state = {k: v.clone() for k, v in self.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if current_metric < global_best_metric:
                    global_best_metric = current_metric
                    global_best_model_state = {k: v.clone() for k, v in self.state_dict().items()}

            if epochs_no_improve >= patience:
                print(f"No improvement for {patience} consecutive epochs; early termination of Phase 1.")
                break

        if phase1_best_model_state is not None:
            self.load_state_dict(phase1_best_model_state)
            print("The best model for Phase 1 has been loaded.")

        print("Estimating the order of the variables...")
        final_A = self.get_effective_A(is_training=False).detach()
        self.order = self.compute_order_from_A(final_A)
        self.hard_mask = self.generate_hard_mask(self.order)
        self.use_order_mask = True
        if self.config.get('use_gumbel', False):
            self.gumbel_tau = tau_min

        tau_start = 1.0
        tau_end = self.config.get('tau_end', 0.0)

        for epoch in range(phase2_epochs):
            self.tau = tau_start + (tau_end - tau_start) * (epoch / max(phase2_epochs - 1, 1))
            self.train()
            total_loss = 0
            valid_batches = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                loss, _ = self.forward(batch, total_T)
                if torch.isnan(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                valid_batches += 1

            avg_train_loss = total_loss / max(valid_batches, 1)

            if self.dag_matrix is not None:
                exp_A = self.get_effective_A(is_training=False).detach().cpu().numpy()
                _, est_causal_matrix = find_optimal_threshold_and_matrix_curve(exp_A, self.dag_matrix)
                TP, FN, TN, FP, PRE, REC, F1, g_score, shd = metric_dag_res(est_causal_matrix, self.dag_matrix)
                current_metric = F1

                if current_metric > global_best_metric:
                    global_best_metric = current_metric
                    global_best_model_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                val_loss = self.evaluate(val_loader, total_T)
                current_metric = val_loss
                print(f"Epoch {phase1_epochs + epoch + 1}, tau={self.tau:.3f}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

                if current_metric < global_best_metric:
                    global_best_metric = current_metric
                    global_best_model_state = {k: v.clone() for k, v in self.state_dict().items()}

        if global_best_model_state is not None:
            self.load_state_dict(global_best_model_state)
            metric_type = "F1" if self.dag_matrix is not None else "Val Loss"
            print(
                f"The model with the best {metric_type} performance across the entire training process (two phases) has been loaded (Score: {global_best_metric:.4f}).")

        final_A_np = self.get_effective_A(is_training=False).detach().cpu().numpy()
        np.fill_diagonal(final_A_np, 0)
        return final_A_np


def CDTES_Func(alarm_data, topo_matrix, dag_matrix, dataset_name=None):
    config = DATASET_CONFIG_MAP[dataset_name]

    set_seed(2025)
    num_event_types = alarm_data['alarm_id'].nunique()
    num_devices = topo_matrix.shape[0]

    event_pairs, total_T = prepare_event_pairs(alarm_data, topo_matrix, k_hop=config.get('k_hop', 2))
    print(f"A total of {len(event_pairs)} target event samples were generated, with a total duration of {total_T:.2f}")

    if len(event_pairs) == 0:
        raise RuntimeError("No samples were generated. Please check the data or adjust k_hop.")

    random.shuffle(event_pairs)
    split_ratio = config.get('split_ratio', 1.0)
    split_idx = int(split_ratio * len(event_pairs))

    train_pairs = event_pairs[:split_idx] if split_ratio < 1.0 else event_pairs
    val_pairs = event_pairs[split_idx:] if split_ratio < 1.0 else event_pairs

    print(f"TRAINING SAMPLES: {len(train_pairs)}")

    train_dataset = EventPairDataset(train_pairs, num_event_types)
    val_dataset = EventPairDataset(val_pairs, num_event_types)
    val_grouped = group_event_pairs(val_pairs)

    batch_size = config.get('batch_size', 32)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=collate_fn, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                                             num_workers=0)

    model = TACDGC(num_event_types, num_devices, topo_matrix, dag_matrix, config).to(device)

    causal_matrix = model.train_model(train_loader, val_loader, val_grouped, total_T)
    return causal_matrix
