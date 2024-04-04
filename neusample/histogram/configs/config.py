default_options = {
    # dataset config
    'batch_size':{
        'type': int,
        'default': 512//2
    },
    'wi_num':{
        'type': int,
        'default': 1024,
    },
    'lobe_size':{
        'type': int,
        'default': 64
    },
    'dataset_length': {
        'type': int,
        'nargs': 2,
        'default': [64,16],
    },
    'num_workers': {
        'type': int,
        'default': 18
    },
    
    'neumip_path': {
        'type': str,
        'default': 'outputs/tortoise_shell.pth' #tortoise_shell elephant_leather_chainmail_emboss victorian_wall_fabric_nodisp
    },
    
    # base 2d
    'base_mlp':{
        'type': int,
        'nargs': 4,
        'default': [2,8+2,32,2,16,4,4]
    },
    
    # nis mlp
    'nis_mlp': {
        'type': int,
        'nargs': 4,
        # layer num, C, D ,encode
        'default': [2, 32, 3, 4],
    },

    # optimizer config
    'optimizer': {
        'type': str,
        'choices': ['SGD', 'Ranger', 'Adam'],
        'default': 'Adam'
    },
    'learning_rate': {
        'type': float,
        'default': 1e-3
    },
    'weight_decay': {
        'type': float,
        'default': 0
    },

    'scheduler_rate':{
        'type': float,
        'default': 0.5
    },
    'milestones':{
        'type': int,
        'nargs': '*',
        'default': [1000] # never used
    },
    
    
    # rendering config
    'white_back': {
        'type': bool,
        'default': False
    }
}
