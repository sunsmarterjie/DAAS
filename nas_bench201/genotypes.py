from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'avg_pool_3x3',
    'skip_connect',
    'conv_1x1',
    'conv_3x3',
]
