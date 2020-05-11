from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer'
# current = 'tester_basic'

mod = '"sta00"' # nothing; builder
mod = '"sta01"' # just prep and return
mod = '"sta02"' # again, fewer prints
mod = '"sta03"' # run feat3d forward; drop the sparse stuff
mod = '"sta04"' # really run it
mod = '"sta05"' # again
mod = '"sta06"' # warp; show altfeat
mod = '"sta07"' # ensure either ==1 or a==b
mod = '"sta08"' # try emb
mod = '"sta09"' # train a while
mod = '"sta10"' # 
mod = '"sta11"' # show altfeat input
mod = '"sta12"' # 
mod = '"sta13"' # train occ
mod = '"sta14"' # move things to R
mod = '"sta14"' # do view
mod = '"sta15"' # encode in X0
mod = '"sta16"' # 
mod = '"sta17"' # show rgb_camX1, so i can understand the inbound idea better
mod = '"sta18"' # show inbound separately
mod = '"sta19"' # allow 0 to 32m
mod = '"sta20"' # builder
mod = '"sta21"' # show occ_memXs
mod = '"sta22"' # wider bounds please
mod = '"sta23"' # properly combine bounds with centorid
mod = '"sta24"' # train a hwile
mod = '"sta25"' # same but encode in Xs and warp to R then X0
mod = '"sta26"' # use resnet3d  
mod = '"sta27"' # skipnet; randomize the centroid a bit
mod = '"sta28"' # wider rand, and inbound check
mod = '"sta29"' # handle the false return
mod = '"sta30"' # add emb2d
mod = '"sta31"' # freeze the slow model
mod = '"sta32"' # 2d parts
mod = '"sta33"' # fewer prints
mod = '"sta34"' # nice suffixes; JUST 2d learning
mod = '"sta35"' # fix bug
mod = '"sta36"' # better summ suffix
mod = '"sta37"' # tell me about neg pool size
mod = '"sta38"' # fix small bug in the hyp lettering
mod = '"sta39"' # cleaned up hyps
mod = '"sta40"' # weak smooth coeff on feats
mod = '"sta41"' # run occnet on altfeat instead

############## define experiment ##############

exps['builder'] = [
    'carla_static', # mode
    'carla_multiview_10_data', # dataset
    'carla_bounds', 
    '3_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_feat3d',
    # 'train_occ',
    # 'train_view',
    # 'train_emb2d',
    # 'train_emb3d',
    'log1',
]
exps['trainer'] = [
    'carla_static', # mode
    'carla_multiview_train_data', # dataset
    'carla_bounds', 
    '300k_iters',
    'lr3',
    'B2',
    'train_feat3d',
    'train_emb3d',
    'train_occ',
    'train_view',
    'train_feat2d',
    'train_emb2d',
    'log500',
]

############## net configs ##############

groups['train_feat2d'] = [
    'do_feat2d = True',
    'feat2d_dim = 32',
    # 'feat2d_smooth_coeff = 0.1',
]
groups['train_feat3d'] = [
    'do_feat3d = True',
    'feat3d_dim = 32',
    'feat3d_smooth_coeff = 0.01',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 0.01',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
]
groups['train_emb2d'] = [
    'do_emb2d = True',
    # 'emb2d_smooth_coeff = 0.01',
    'emb2d_ce_coeff = 1.0',
    'emb2d_l2_coeff = 0.1',
    'emb2d_mindist = 32.0',
    'emb2d_num_samples = 4',
    # 'do_view = True',
    # 'view_depth = 32',
    # 'view_l1_coeff = 1.0',
]
groups['train_emb3d'] = [
    'do_emb3d = True',
    'emb3d_ce_coeff = 1.0',
    # 'emb_3d_l2_coeff = 0.1',
    # 'emb_3d_mindist = 16.0',
    'emb3d_num_samples = 2',
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

groups['carla_multiview_10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mags7i3v"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_val_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "mags7i3v"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
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
