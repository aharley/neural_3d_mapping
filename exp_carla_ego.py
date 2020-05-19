from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'debugger'
current = 'trainer'

mod = '"eg00"' # nothing; builder
mod = '"eg01"' # deleted junk
mod = '"eg02"' # added hyps
mod = '"eg03"' # train a while
mod = '"eg04"' # 1 scale
mod = '"eg05"' # no synth
mod = '"eg06"' # consec=True
mod = '"eg07"' # comment out the synth part < ok. but this npz has no motion
mod = '"eg08"' # second file < a bit jumpier than i would like...
mod = '"eg09"' # S = 3
mod = '"eg10"' # make my own thing; assert S==2 < ok, much cleaner, but still jumpy
mod = '"eg11"' # cleaned up summs
mod = '"eg12"' # cleaned up summs; include the occ transform 
mod = '"eg13"' # removed the warp loss
mod = '"eg14"' # add summ of the gt
mod = '"eg15"' # fix the hyps
mod = '"eg16"' # renamed DHW as ZYX
mod = '"eg17"' # same, fewer prints
mod = '"eg18"' # feed rgbd input
mod = '"eg19"' # cleaned up
mod = '"eg20"' # train a while

############## exps ##############

exps['builder'] = [
    'carla_ego', # mode
    'carla_traj_10_data', # dataset
    'carla_bounds', 
    '3_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_feat3d',
    'train_ego',
    'log1',
]
exps['debugger'] = [
    'carla_ego', # mode
    'carla_traj_1_data', # dataset
    'carla_bounds', 
    '1k_iters',
    'lr4',
    'B1',
    'train_feat3d',
    'train_ego',
    'no_shuf',
    'log10',
]
exps['trainer'] = [
    'carla_ego', # mode
    'carla_traj_train_data', # dataset
    'carla_bounds', 
    '100k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    'train_ego',
    'log50',
]

############## groups ##############

groups['carla_ego'] = ['do_carla_ego = True']

groups['train_feat3d'] = [
    'do_feat3d = True',
    'feat3d_dim = 32',
]
groups['train_ego'] = [
    'do_ego = True',
    'ego_t_l2_coeff = 1.0',
    'ego_deg_l2_coeff = 1.0',
    'ego_num_scales = 2',
    'ego_num_rots = 11',
    'ego_max_deg = 4.0',
    'ego_max_disp_z = 2',
    'ego_max_disp_y = 1',
    'ego_max_disp_x = 2',
    'ego_synth_prob = 0.0',
]

############## datasets ##############

# dims for mem
SIZE = 32
Z = int(SIZE*4)
Y = int(SIZE*1)
X = int(SIZE*4)
K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 2
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"

groups['carla_traj_1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "taqs100i2one"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traj_10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "taqs100i2ten"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traj_train_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "taqs100i2t"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]

############## verify and execute ##############

def _verify_(s):
    varname, eq, val = s.split(' ')
    assert varname in globals()
    assert eq == '='
    assert type(s) is type('')

print(current)
assert current in exps
for group in exps[current]:
    print("  " + group)
    assert group in groups
    for s in groups[group]:
        print("    " + s)
        _verify_(s)
        exec(s)

s = "mod = " + mod
_verify_(s)

exec(s)
