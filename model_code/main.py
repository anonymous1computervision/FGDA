from solver import mySolver

config = {
    'batch_size': [128, 48],
    'epoch': 100,
    'lr': 1e-4,
    'weight_decay': 5e-4,
    'momentum': 0.9,

    'lr_f': 10,
    'lr_c': 5,
    'lr_d': 2,

    'FC/D': 10,

    'coral': 0,
    'mmd': 0,
    'domain': 1.0,
    'class': 0.0,
    'ori': 0.3,

    'spatial': True,
    'concat': False,
    'spatial_dis': False,

    'random': True,

    'source': 'sku',
    'target': 'shelf',

    'domainloss': 2,
    'fadomainloss': 1

}

solver = mySolver(config)
solver.train()
