import os
# from munch import Munch

H = 240 # height
W = 320 # width

Z = 128
Y = 64
X = 128
Z_val = 128
Y_val = 64
X_val = 128
Z_test = 128
Y_test = 64
X_test = 128

PH = int(128/4)
PW = int(384/4)

ZY = 32
ZX = 32
ZZ = 32

N = 50 # number of boxes produced by the rcnn (not all are good)
K = 1 # number of boxes to actually use
S = 2 # seq length
S_test = 3 # seq length
T = 256 # height & width of birdview map
V = 100000 # num velodyne points

# metric bounds of mem space 
XMIN = -16.0 # right (neg is left)
XMAX = 16.0 # right
YMIN = -1.0 # down (neg is up)
YMAX = 3.0 # down
ZMIN = 2.0 # forward
ZMAX = 34.0 # forward
XMIN_val = -16.0 # right (neg is left)
XMAX_val = 16.0 # right
YMIN_val = -1.0 # down (neg is up)
YMAX_val = 3.0 # down
ZMIN_val = 2.0 # forward
ZMAX_val = 34.0 # forward
XMIN_test = -16.0 # right (neg is left)
XMAX_test = 16.0 # right
YMIN_test = -1.0 # down (neg is up)
YMAX_test = 3.0 # down
ZMIN_test = 2.0 # forward
ZMAX_test = 34.0 # forward
FLOOR = 2.65 # ground (2.65m downward from the cam)
CEIL = (FLOOR-2.0) # 

#----------- loading -----------#

do_include_summs = False
do_include_vis = True
do_test = False
do_export_vis = False
do_export_stats = False
do_export_inds = False


emb2d_init = ""
feat2d_init = ""
feat3d_init = ""
flow_init = ""
occ_init = ""
view_init = ""
det_init = ""
ego_init = ""

total_init = ""
reset_iter = False

do_freeze_emb2d = False
do_freeze_feat2d = False
do_freeze_feat3d = False
do_freeze_occ = False
do_freeze_view = False
do_freeze_flow = False
do_freeze_ego = False
do_freeze_det = False
do_resume = False
do_profile = False

# by default, only backprop on "train" iters
backprop_on_train = True
backprop_on_val = False
backprop_on_test = False

# eval mode: save npys
do_eval_map = False
do_eval_recall = False # keep a buffer and eval recall within it
do_save_embs = False
do_save_ego = False
do_save_vis = False
do_save_outputs = False

#----------- augs -----------#
# do_aug2d = False
# do_aug3d = False
do_aug_color = False
do_time_flip = False
do_horz_flip = False
do_synth_rt = False
do_synth_nomotion = False
do_piecewise_rt = False
do_sparsify_pointcloud = 0 # choose a number here, for # pts to use

#----------- net design -----------#
# by default, run nothing
do_emb2d = False
do_emb3d = False
do_feat2d = False
do_feat3d = False
do_occ = False
do_view = False
do_flow = False
do_ego = False
do_det = False

#----------- general hypers -----------#
lr = 0.0

#----------- emb hypers -----------#
emb2d_ml_coeff = 0.0
emb3d_ml_coeff = 0.0
emb2d_l2_coeff = 0.0
emb3d_l2_coeff = 0.0
emb2d_mindist = 0.0
emb3d_mindist = 0.0
emb2d_num_samples = 0
emb3d_num_samples = 0
emb2d_ce_coeff = 0.0
emb3d_ce_coeff = 0.0

#----------- feat3d hypers -----------#
feat3d_dim = 32

#----------- feat2d hypers -----------#
feat2d_smooth_coeff = 0.0
feat2d_dim = 8

#----------- occ hypers -----------#
occ_coeff = 0.0
occ_smooth_coeff = 0.0

#----------- view hypers -----------#
view_depth = 64
view_accu_render = False
view_accu_render_unps = False
view_accu_render_gt = False
view_pred_embs = False
view_pred_rgb = False
view_l1_coeff = 0.0

#----------- det hypers -----------#
det_anchor_size = 12.0
det_prob_coeff = 0.0
det_reg_coeff = 0.0

#----------- flow hypers -----------#
flow_warp_coeff = 0.0
flow_warp_g_coeff = 0.0
flow_cycle_coeff = 0.0
flow_smooth_coeff = 0.0
flow_l1_coeff = 0.0
flow_l2_coeff = 0.0
# flow_synth_l1_coeff = 0.0
# flow_synth_l2_coeff = 0.0
flow_do_synth_rt = False
flow_heatmap_size = 4

#----------- ego hypers -----------#
ego_use_gt = False
ego_use_precomputed = False
ego_rtd_coeff = 0.0
ego_rta_coeff = 0.0
ego_traj_coeff = 0.0
ego_warp_coeff = 0.0

#----------- mod -----------#

mod = '""'

############ slower-to-change hyperparams below here ############

## logging
log_freq_train = 100
log_freq_val = 100
log_freq_test = 100
snap_freq = 10000

max_iters = 10000
shuffle_train = True
shuffle_val = True
shuffle_test = True

trainset_format = 'seq'
valset_format = 'seq'
testset_format = 'seq'
# should the seqdim be taken in consecutive order
trainset_consec = True
valset_consec = True
testset_consec = True

trainset_seqlen = 2
valset_seqlen = 2
testset_seqlen = 2

trainset_batch_size = 2
valset_batch_size = 1
testset_batch_size = 1

dataset_name = ""
seqname = ""
ind_dataset = ''

trainset = ""
valset = ""
testset = ""

dataset_location = ""
dataset_filetype = "npz"

# mode selection
do_carla_static = False
do_carla_time = False
do_carla_reloc = False
do_carla_obj = False
do_carla_focus = False
do_carla_track = False
do_carla_siamese = False
do_carla_vsiamese = False
do_carla_rsiamese = False
do_carla_msiamese = False
do_carla_ssiamese = False
do_carla_csiamese = False
do_carla_genocc = False
do_carla_gengray = False
do_carla_gen2dvq = False
do_carla_sigen2d = False
do_carla_sigen2d = False
do_carla_moc = False
do_carla_don = False
do_kitti_don = False
do_kitti_moc = False
do_carla_zoom = False
do_kitti_zoom = False
do_kitti_siamese = False
do_carla_ml = False
do_carla_bottle = False
do_carla_pretty = False
do_carla_bench = False
do_carla_reject = False
do_carla_auto = False
do_carla_ret = False
do_carla_vqrgb = False
do_carla_vq3drgb = False
do_clevr_vq3drgb = False
do_clevr_gen3dvq = False
do_carla_gen3dvq = False
do_carla_precompute = False
do_carla_propose = False
do_carla_det = False
do_intphys_det = False
do_intphys_forecast = False
do_carla_forecast = False
do_carla_pipe = False
do_intphys_test = False
do_mujoco_offline = False
do_carla_pwc = False

identifier_self_define = ""
############ rev up the experiment ############

mode = os.environ["MODE"]
print('os.environ mode is %s' % mode)
if mode=="CARLA_STATIC":
    exec(compile(open('exp_carla_static.py').read(), 'exp_carla_static.py', 'exec'))
elif mode=="CARLA_FLO":
    exec(compile(open('exp_carla_flo.py').read(), 'exp_carla_flo.py', 'exec'))
elif mode=="CARLA_TIME":
    exec(compile(open('exp_carla_time.py').read(), 'exp_carla_time.py', 'exec'))
elif mode=="CARLA_RELOC":
    exec(compile(open('exp_carla_reloc.py').read(), 'exp_carla_reloc.py', 'exec'))
elif mode=="CARLA_OBJ":
    exec(compile(open('exp_carla_obj.py').read(), 'exp_carla_obj.py', 'exec'))
elif mode=="CARLA_FOCUS":
    exec(compile(open('exp_carla_focus.py').read(), 'exp_carla_focus.py', 'exec'))
elif mode=="CARLA_TRACK":
    exec(compile(open('exp_carla_track.py').read(), 'exp_carla_track.py', 'exec'))
elif mode=="CARLA_SIAMESE":
    exec(compile(open('exp_carla_siamese.py').read(), 'exp_carla_siamese.py', 'exec'))
elif mode=="CARLA_VSIAMESE":
    exec(compile(open('exp_carla_vsiamese.py').read(), 'exp_carla_vsiamese.py', 'exec'))
elif mode=="CARLA_RSIAMESE":
    exec(compile(open('exp_carla_rsiamese.py').read(), 'exp_carla_rsiamese.py', 'exec'))
elif mode=="CARLA_MSIAMESE":
    exec(compile(open('exp_carla_msiamese.py').read(), 'exp_carla_msiamese.py', 'exec'))
elif mode=="CARLA_SSIAMESE":
    exec(compile(open('exp_carla_ssiamese.py').read(), 'exp_carla_ssiamese.py', 'exec'))
elif mode=="CARLA_CSIAMESE":
    exec(compile(open('exp_carla_csiamese.py').read(), 'exp_carla_csiamese.py', 'exec'))
elif mode=="CARLA_GENOCC":
    exec(compile(open('exp_carla_genocc.py').read(), 'exp_carla_genocc.py', 'exec'))
elif mode=="CARLA_GENGRAY":
    exec(compile(open('exp_carla_gengray.py').read(), 'exp_carla_gengray.py', 'exec'))
elif mode=="CARLA_VQRGB":
    exec(compile(open('exp_carla_vqrgb.py').read(), 'exp_carla_vqrgb.py', 'exec'))
elif mode=="CARLA_VQ3dRGB":
    exec(compile(open('exp_carla_vq3drgb.py').read(), 'exp_carla_vq3drgb.py', 'exec'))
elif mode=="CARLA_MOC":
    exec(compile(open('exp_carla_moc.py').read(), 'exp_carla_moc.py', 'exec'))
elif mode=="CARLA_DON":
    exec(compile(open('exp_carla_don.py').read(), 'exp_carla_don.py', 'exec'))
elif mode=="KITTI_DON":
    exec(compile(open('exp_kitti_don.py').read(), 'exp_kitti_don.py', 'exec'))
elif mode=="KITTI_MOC":
    exec(compile(open('exp_kitti_moc.py').read(), 'exp_kitti_moc.py', 'exec'))
elif mode=="CARLA_ZOOM":
    exec(compile(open('exp_carla_zoom.py').read(), 'exp_carla_zoom.py', 'exec'))
elif mode=="KITTI_ZOOM":
    exec(compile(open('exp_kitti_zoom.py').read(), 'exp_kitti_zoom.py', 'exec'))
elif mode=="KITTI_SIAMESE":
    exec(compile(open('exp_kitti_siamese.py').read(), 'exp_kitti_siamese.py', 'exec'))
elif mode=="CARLA_ML":
    exec(compile(open('exp_carla_ml.py').read(), 'exp_carla_ml.py', 'exec'))
elif mode=="CARLA_BOTTLE":
    exec(compile(open('exp_carla_bottle.py').read(), 'exp_carla_bottle.py', 'exec'))
elif mode=="CARLA_PRETTY":
    exec(compile(open('exp_carla_pretty.py').read(), 'exp_carla_pretty.py', 'exec'))
elif mode=="CARLA_BENCH":
    exec(compile(open('exp_carla_bench.py').read(), 'exp_carla_bench.py', 'exec'))
elif mode=="CARLA_REJECT":
    exec(compile(open('exp_carla_reject.py').read(), 'exp_carla_reject.py', 'exec'))
elif mode=="CARLA_AUTO":
    exec(compile(open('exp_carla_auto.py').read(), 'exp_carla_auto.py', 'exec'))
elif mode=="CARLA_RET":
    exec(compile(open('exp_carla_ret.py').read(), 'exp_carla_ret.py', 'exec'))
elif mode=="CLEVR_VQ3d":
    exec(compile(open('exp_clevr_vq3d.py').read(), 'exp_clevr_vq3d.py', 'exec'))
elif mode=="CLEVR_GEN3dVQ":
    exec(compile(open('exp_clevr_gen3dvq.py').read(), 'exp_clevr_gen3dvq.py', 'exec'))
elif mode=="CARLA_GEN3dVQ":
    exec(compile(open('exp_carla_gen3dvq.py').read(), 'exp_carla_gen3dvq.py', 'exec'))
elif mode=="CARLA_PRECOMPUTE":
    exec(compile(open('exp_carla_precompute.py').read(), 'exp_carla_precompute.py', 'exec'))
elif mode=="CARLA_PROPOSE":
    exec(compile(open('exp_carla_propose.py').read(), 'exp_carla_propose.py', 'exec'))
elif mode=="CARLA_DET":
    exec(compile(open('exp_carla_det.py').read(), 'exp_carla_det.py', 'exec'))
elif mode=="INTPHYS_DET":
    exec(compile(open('exp_intphys_det.py').read(), 'exp_intphys_det.py', 'exec'))
elif mode=="INTPHYS_FORECAST":
    exec(compile(open('exp_intphys_forecast.py').read(), 'exp_intphys_forecast.py', 'exec'))
elif mode=="CARLA_FORECAST":
    exec(compile(open('exp_carla_forecast.py').read(), 'exp_carla_forecast.py', 'exec'))
elif mode=="CARLA_PIPE":
    exec(compile(open('exp_carla_pipe.py').read(), 'exp_carla_pipe.py', 'exec'))
elif mode=="INTPHYS_TEST":
    exec(compile(open('exp_intphys_test.py').read(), 'exp_intphys_test.py', 'exec'))
elif mode=="MUJOCO_OFFLINE":
    exec(compile(open('exp_mujoco_offline.py').read(), 'exp_mujoco_offline.py', 'exec'))
elif mode=="CARLA_PWC":
    exec(compile(open('exp_carla_pwc.py').read(), 'exp_carla_pwc.py', 'exec'))
elif mode=="CUSTOM":
    exec(compile(open('exp_custom.py').read(), 'exp_custom.py', 'exec'))
else:
    assert(False) # what mode is this?

############ make some final adjustments ############

if not do_mujoco_offline:
    trainset_path = "%s/%s.txt" % (dataset_location, trainset)
    valset_path = "%s/%s.txt" % (dataset_location, valset)
    testset_path = "%s/%s.txt" % (dataset_location, testset)
else:
    trainset_path = "%s/%s.npy" % (dataset_location, trainset)
    valset_path = "%s/%s.npy" % (dataset_location, valset)
    testset_path = "%s/%s.npy" % (dataset_location, testset)

data_paths = {}
data_paths['train'] = trainset_path
data_paths['val'] = valset_path
data_paths['test'] = testset_path

set_nums = {}
set_nums['train'] = 0
set_nums['val'] = 1
set_nums['test'] = 2

set_names = ['train', 'val', 'test']

log_freqs = {}
log_freqs['train'] = log_freq_train
log_freqs['val'] = log_freq_val
log_freqs['test'] = log_freq_test

shuffles = {}
shuffles['train'] = shuffle_train
shuffles['val'] = shuffle_val
shuffles['test'] = shuffle_test

data_formats = {}
data_formats['train'] = trainset_format
data_formats['val'] = valset_format
data_formats['test'] = testset_format

data_consecs = {}
data_consecs['train'] = trainset_consec
data_consecs['val'] = valset_consec
data_consecs['test'] = testset_consec

seqlens = {}
seqlens['train'] = trainset_seqlen
seqlens['val'] = valset_seqlen
seqlens['test'] = testset_seqlen

batch_sizes = {}
batch_sizes['train'] = trainset_batch_size
batch_sizes['val'] = valset_batch_size
batch_sizes['test'] = testset_batch_size


############ autogen a name; don't touch any hypers! ############

def strnum(x):
    s = '%g' % x
    if '.' in s:
        s = s[s.index('.'):]
    return s


if do_test:
    name = "%02d_s%d" % (testset_batch_size, S_test)
    name += "_m%dx%dx%d" % (Z_test, Y_test, X_test)
else:
    name = "%02d_s%d" % (trainset_batch_size, S)
    if do_feat3d:
        name += "_m%dx%dx%d" % (Z, Y, X)
    
if do_view or do_emb2d:
    name += "_p%dx%d" % (PH,PW)

if lr > 0.0:
    lrn = "%.1e" % lr
    # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1]
    name += "_%s" % lrn

if do_feat2d:
    name += "_F2"
    if do_freeze_feat2d:
        name += "f"
    coeffs = [
        feat2d_dim,
        feat2d_smooth_coeff,
    ]
    prefixes = [
        "d",
        "s",
    ]
    for l_, l in enumerate(coeffs):
        if l > 0:
            name += "_%s%s" % (prefixes[l_],strnum(l))
    
if do_feat3d:
    name += "_F3"
    if do_freeze_feat3d:
        name += "f"
    coeffs = [
        feat3d_dim,
    ]
    prefixes = [
        "d",
    ]
    for l_, l in enumerate(coeffs):
        if l > 0:
            name += "_%s%s" % (prefixes[l_],strnum(l))

if do_ego:
    name += "_G"
    if ego_use_gt:
        name += "gt"
    elif ego_use_precomputed:
        name += "pr"
    else:
        if do_freeze_ego:
            name += "f"
        else:
            ego_coeffs = [
                ego_rtd_coeff,
                ego_rta_coeff,
                ego_traj_coeff,
                ego_warp_coeff,
            ]
            ego_prefixes = [
                "rtd",
                "rta",
                "t",
                "w",
            ]
            for l_, l in enumerate(ego_coeffs):
                if l > 0:
                    name += "_%s%s" % (ego_prefixes[l_],strnum(l))

if do_occ:
    name += "_O"
    if do_freeze_occ:
        name += "f"
    occ_coeffs = [
        occ_coeff,
        occ_smooth_coeff,
    ]
    occ_prefixes = [
        "c",
        "s",
    ]
    for l_, l in enumerate(occ_coeffs):
        if l > 0:
            name += "_%s%s" % (occ_prefixes[l_],strnum(l))

if do_view:
    name += "_V"
    if view_pred_embs:
        name += "e"
    if view_pred_rgb:
        name += "r"
    if do_freeze_view:
        name += "f"
    view_coeffs = [
        view_depth,
        view_l1_coeff,
    ]
    view_prefixes = [
        "d",
        "e",
    ]
    for l_, l in enumerate(view_coeffs):
        if l > 0:
            name += "_%s%s" % (view_prefixes[l_],strnum(l))

if do_det:
    name += "_D"
    name += "%d" % det_anchor_size
    if do_freeze_det:
        name += "f"
    det_coeffs = [
        det_prob_coeff,
        det_reg_coeff,
    ]
    det_prefixes = [
        "p",
        "r",
    ]
    for l_, l in enumerate(det_coeffs):
        if l > 0:
            name += "_%s%s" % (det_prefixes[l_],strnum(l))
            
if do_emb2d:
    name += "_E2"
    if do_freeze_emb2d:
        name += "f"
    coeffs = [
        emb2d_ml_coeff,
        emb2d_l2_coeff,
        emb2d_num_samples,
        emb2d_mindist,
        emb2d_ce_coeff,
    ]
    prefixes = [
        "m",
        "e",
        "n",
        "d",
        "c",
    ]
    for l_, l in enumerate(coeffs):
        if l > 0:
            name += "_%s%s" % (prefixes[l_],strnum(l))
            
if do_emb3d:
    name += "_E3"
    coeffs = [
        emb3d_ml_coeff,
        emb3d_l2_coeff,
        emb3d_num_samples,
        emb3d_mindist,
        emb3d_ce_coeff,
    ]
    prefixes = [
        "m",
        "e",
        "n",
        "d",
        "c",
    ]
    for l_, l in enumerate(coeffs):
        if l > 0:
            name += "_%s%s" % (prefixes[l_],strnum(l))
            
if do_flow:
    name += "_F"
    if do_freeze_flow:
        name += "f"
    else:
        flow_coeffs = [flow_heatmap_size,
                       flow_warp_coeff,
                       flow_warp_g_coeff,
                       flow_cycle_coeff,
                       flow_smooth_coeff,
                       flow_l1_coeff,
                       flow_l2_coeff,
                       # flow_synth_l1_coeff,
                       # flow_synth_l2_coeff,
        ]
        flow_prefixes = ["h",
                         "w",
                         "g",
                         "c",
                         "s",
                         "e",
                         "f",
                         # "y",
                         # "x",
        ]
        for l_, l in enumerate(flow_coeffs):
            if l > 0:
                name += "_%s%s" % (flow_prefixes[l_],strnum(l))

##### end model description

# add some training data info

sets_to_run = {}
if trainset:
    name = "%s_%s" % (name, trainset)
    sets_to_run['train'] = True
else:
    sets_to_run['train'] = False

if valset:
    name = "%s_%s" % (name, valset)
    sets_to_run['val'] = True
else:
    sets_to_run['val'] = False

if testset:
    name = "%s_%s" % (name, testset)
    sets_to_run['test'] = True
else:
    sets_to_run['test'] = False

sets_to_backprop = {}
sets_to_backprop['train'] = backprop_on_train
sets_to_backprop['val'] = backprop_on_val
sets_to_backprop['test'] = backprop_on_test

if (do_aug_color or
    do_horz_flip or
    do_time_flip or
    do_synth_rt or
    do_piecewise_rt or
    do_synth_nomotion or
    do_sparsify_pointcloud):
    name += "_A"
    if do_aug_color:
        name += "c"
    if do_horz_flip:
        name += "h"
    if do_time_flip:
        name += "t"
    if do_synth_rt:
        assert(not do_piecewise_rt)
        name += "s"
    if do_piecewise_rt:
        assert(not do_synth_rt)
        name += "p"
    if do_synth_nomotion:
        name += "n"
    if do_sparsify_pointcloud:
        name += "v"

if (not shuffle_train) or (not shuffle_val) or (not shuffle_test):
    name += "_ns"

if not do_include_vis:
    name += "_nv"

if do_profile:
    name += "_PR"

if mod:
    name = "%s_%s" % (name, mod)
if len(identifier_self_define) > 0:
    name += ('_' + identifier_self_define)

if do_resume:
    name += '_gt'
    total_init = name

print(name)
