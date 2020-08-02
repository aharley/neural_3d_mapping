from exp_base import *

############## choose an experiment ##############

current = 'det_builder'
current = 'det_trainer'

mod = '"det00"' # go
mod = '"det01"' # rescore with inbound
mod = '"det02"' # show scores
mod = '"det03"' # show bev too
mod = '"det04"' # narrower bounds, to see 
mod = '"det05"' # print scorelists
mod = '"det06"' # rescore actually
mod = '"det07"' # print score no matter what
mod = '"det08"' # float2str
mod = '"det09"' # run feat3d
mod = '"det10"' # really run feat3d
mod = '"det11"' # get axboxlist
mod = '"det12"' # solid centorid
mod = '"det13"' # update lrtlist util
mod = '"det14"' # run detnet
mod = '"det15"' # 
mod = '"det16"' # train a whiel
mod = '"det17"' # bugfix
mod = '"det18"' # return early if score < B/2
mod = '"det19"' # new utils
mod = '"det20"' # B2
mod = '"det21"' # B4
mod = '"det22"' # clean up
mod = '"det23"' # rand centroid
mod = '"det24"' # padding 0
mod = '"det25"' # avoid warping to R0
mod = '"det26"' # scorelist *= inbound
mod = '"det27"' # use scorelist in the vis
mod = '"det28"' # only draw nonzero boxes
mod = '"det29"' # cleaned up
mod = '"det30"' # do not draw 0,1 scores
mod = '"det31"' # cleaned up

############## define experiments ##############

exps['det_builder'] = [
    'carla_det', # mode
    'carla_multiview_10_data', # dataset
    'seqlen1',
    '8-4-8_bounds',
    # '16-8-16_bounds',
    '3_iters',
    # '5k_iters',
    # 'lr3', 
    'train_feat3d',
    'train_det',
    'B1',
    'no_shuf',
    'no_backprop',
    # 'log50',
    'log1',
]
exps['det_trainer'] = [
    'carla_det', # mode
    # 'carla_multiview_10_data', # dataset
    'carla_multiview_train_data', # dataset
    'seqlen1',
    # 'carla_multiview_train_val_data', # dataset
    '16-8-16_bounds',
    # 'carla_16-8-16_bounds_train', 
    # 'carla_16-8-16_bounds_val', 
    '200k_iters',
    'lr3',
    # 'B1',
    'B4',
    'train_feat3d',
    'train_det',
    'log50', 
]

############## group configs ##############

groups['carla_det'] = ['do_carla_det = True']



############## datasets ##############

# DHW for mem stuff
SIZE = 32
Z = int(SIZE*4)
Y = int(SIZE*2)
X = int(SIZE*4)

K = 8 # how many proposals to consider

# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)


# S = 1
# groups['carla_multiview_10_data'] = [
#     'dataset_name = "carla"',
#     'H = %d' % H,
#     'W = %d' % W,
#     'trainset = "mags7i3ten"',
#     'trainset_format = "multiview"', 
#     'trainset_seqlen = %d' % S, 
#     'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
#     'dataset_filetype = "npz"'
# ]
# groups['carla_multiview_train_val_data'] = [
#     'dataset_name = "carla"',
#     'H = %d' % H,
#     'W = %d' % W,
#     'trainset = "mags7i3t"',
#     'trainset_format = "multiview"', 
#     'trainset_seqlen = %d' % S, 
#     'valset = "mags7i3v"',
#     'valset_format = "multiview"', 
#     'valset_seqlen = %d' % S, 
#     'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
#     'dataset_filetype = "npz"'
# ]

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
