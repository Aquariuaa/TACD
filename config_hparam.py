

config_sim = {
    'k_hop': 2, 'type_emb_dim': 16, 'dev_emb_dim': 8,
    'lambda1': 5e-3, 'lambda2': 4e3, 'lr': 1e-2, 'batch_size': 32,
    'phase1_epochs': 25, 'phase2_epochs': 5, 'patience': 25,
    'use_time_decay': False,
    'mu_type': 'device_specific',
    'use_gumbel': False,
    'tau_end': 0.8,
    'split_ratio': 1.0,
    'lr_schedule': {0: 1e-2}
}

config_18V = {
    'k_hop': 2, 'type_emb_dim': 16, 'dev_emb_dim': 8,
    'lambda1': 5e-3, 'lambda2': 1e-5, 'lr': 1e-3, 'batch_size': 32,
    'phase1_epochs': 25, 'phase2_epochs': 5, 'patience': 25,
    'use_time_decay': True,
    'mu_type': 'shared',
    'use_gumbel': False,
    'tau_end': 0.2,
    'split_ratio': 1.0,
    'lr_schedule': {0: 1e-3}
}

config_24V = {
    'k_hop': 2, 'type_emb_dim': 256, 'dev_emb_dim': 128,
    'lambda1': 5e-2, 'lambda2': 1e-5, 'lr': 1e-4, 'batch_size': 32,
    'phase1_epochs': 50, 'phase2_epochs': 5, 'patience': 30,
    'use_time_decay': False,
    'mu_type': 'shared',
    'use_gumbel': False,
    'tau_end': 0.2,
    'split_ratio': 1.0,
    'lr_schedule': {0: 1e-4}
}

config_25V = {
    'k_hop': 2, 'type_emb_dim': 256, 'dev_emb_dim': 128,
    'lambda1': 5e-3, 'lambda2': 1e-5, 'lr': 1e-2, 'batch_size': 512,
    'phase1_epochs': 50, 'phase2_epochs': 20, 'patience': 50,
    'use_time_decay': False,
    'mu_type': 'shared',
    'use_gumbel': True,
    'tau_end': 0.2,
    'split_ratio': 1.0,
    'lr_schedule': {0: 1e-2}
}