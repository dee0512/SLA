layers = {
    'default': {
        'clock': True,
        'clock_dim': 10,
        'previous_action': True
    },

    'MountainCarContinuous-v0': {
        'layers': 3,
        'layer1': [0, 1],
        'layer2': [0],
        'layer3': []
    },

    'Pendulum-v1':{
        'clock_dim': 10,
        'layers': 3,
        'layer1': [0, 1, 2],
        'layer2': [2],
        'layer3': []
    },

    'InvertedPendulum-v2': {
        'layers': 2,
        'layer1': [0, 1],
        'layer2': [],
        'layer3': []
    },

    'Hopper-v2':{
        'clock': False,
        'layers': 2,
        'layer1': [7, 8, 9, 10],
        'layer2': []
    }
}


def get_layers(env_name):
    obj = layers['default']
    for key in layers[env_name].keys():
        obj[key] = layers[env_name][key]
    return obj
