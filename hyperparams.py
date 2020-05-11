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

pb_model = ""
do_include_summs = False
do_include_vis = True
do_test = False
do_export_vis = False
do_export_stats = False
do_export_inds = False


optim_init = ""
vq2d_init = ""
vq3d_init = ""
sigen3d_init = ""
motionreg_init = ""
gen3d_init = ""
pri2d_init = ""
emb2d_init = ""
feat2d_init = ""
feat3d_init = ""
up3d_init = ""
reject3d_init = ""
obj_init = ""
box_init = ""
ort_init = ""
inp_init = ""
traj_init = ""
occ_init = ""
center_init = ""
seg_init = ""
preocc_init = ""
view_init = ""
loc_init = ""
render_init = ""
vis_init = ""
det_init = ""
forecast_init = ""
motioncost_init = ""
motionreg_init = ""
flow_init = ""
loc_init = ""
ego_init = ""
match_init = ""
conf_init = ""
translation_init = ""
rigid_init = ""
robust_init = ""

total_init = ""
reset_iter = False

do_freeze_linclass = False
do_freeze_match = False
do_freeze_conf = False
do_freeze_motionreg = False
do_freeze_emb2d = False
do_freeze_feat2d = False
do_freeze_feat3d = False
do_freeze_up3d = False
do_freeze_reject3d = False
do_freeze_genocc = False
do_freeze_gengray = False
do_freeze_gen2dvq = False
do_freeze_gen3d = False
do_freeze_sigen2d = False
do_freeze_sigen3d = False
do_freeze_vq2d = False
do_freeze_vq3d = False
do_freeze_obj = False
do_freeze_box = False
do_freeze_ort = False
do_freeze_inp = False
do_freeze_traj = False
do_freeze_occ = False
do_freeze_center = False
do_freeze_seg = False
do_freeze_preocc = False
do_freeze_view = False
do_freeze_render = False
do_freeze_vis = False
do_freeze_flow = False
do_freeze_loc = False
do_freeze_ego = False
do_freeze_det = False
do_freeze_pri2d = False
do_freeze_pri3d = False
do_freeze_rpo2d = False
do_freeze_rpo3d = False
do_freeze_forecast = False
do_freeze_motioncost = False
do_freeze_motionreg = False
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
# run nothing
do_linclass = False
do_emb2d = False
do_tri2d = False
do_col2d = False
do_emb3d = False
do_feat3d = False
do_up3d = False
do_reject3d = False
do_feat2d = False
do_genocc = False
do_gengray = False
do_gen2dvq = False
do_gen3d = False
do_sigen2d = False
do_sigen3d = False
do_vq2d = False
do_vq3d = False
do_match = False
do_conf = False
do_translation = False
do_rigid = False
do_robust = False
do_obj = False
do_box = False
do_ort = False
do_inp = False
do_traj = False
do_occ = False
do_center = False
do_seg = False
do_preocc = False
do_view = False
do_render = False
do_flow = False
do_loc = False
do_ego = False
do_vis = False
do_det = False
do_pri2d = False
do_pri3d = False
do_rpo2d = False
do_rpo3d = False
do_forecast = False
do_motioncost = False
do_motionreg = False
do_moc2d = False
do_moc3d = False


#----------- hardvis hypers -----------#

do_hard_vis = False
box_size_xz = 16
box_size_y = 16
box_size_vis = 80
use_random_boxes = False
break_constraint = False
do_debug= False
eval_recall_log_freq = 50

# for custom_max in ["max"]:
#     exec(f"{custom_max} = Munch()")
#     exec(f"{custom_max}.do = False")
#     exec(f"{custom_max}.tdata = False")
#     exec(f"{custom_max}.B = 1")
#     exec(f"{custom_max}.max_iters = 10")
#     exec(f"{custom_max}.log_freq =100")
#     exec(f"{custom_max}.shuffle = False")
#     exec(f"{custom_max}.g_max_iters = 0")
#     exec(f"{custom_max}.p_max_iters = 1")
#     exec(f"{custom_max}.predicted_matching = False")
#     exec(f"{custom_max}.num_patches_per_emb = 10")
#     exec(f"{custom_max}.hardmining = False")
#     exec(f"{custom_max}.tripleLossThreshold = 0.8")
#     exec(f"{custom_max}.max_epochs = 1")
#     exec(f"{custom_max}.hardmining_gt = False")    
#     exec(f"{custom_max}.searchRegion = 2") #size of the candidate proposal region
#     exec(f"{custom_max}.shouldResizeToRandomScale = False") #should scale randomly or not
#     exec(f"{custom_max}.margin = 3") #margin  while selecting the candidate proposals (select everything within the margin) (should be half of valid Region)!
#     exec(f"{custom_max}.numRetrievalsForEachQueryEmb = 20") # the number of retrieval in the pool to consider while finding candidate prooposal region
#     exec(f"{custom_max}.topK = 10") #the number of top candidate proposals to use based on simple matching (default value used in paper is 10 but in our case it should be less than numRetrievalsForEachQueryEmb or it will give a bug!)
#     exec(f"{custom_max}.nbImgEpoch = 200") #the number of top pos pairs to use based on selection verification (default is 200)
#     exec(f"{custom_max}.trainRegion = 6") #the regions distance from the center matches to train on (works better if a be  bit bigger than valid Region as mentioned in the paper) 
#     exec(f"{custom_max}.validRegion = 6") #region to consider while trying to validate retreival (should be double of margin!)
#     exec(f"{custom_max}.visualizeHardMines = False") #region to consider while trying to validate retreival (should be double of margin!)
#     exec(f"{custom_max}.hard_moc = False")
#     exec(f"{custom_max}.hard_moc_qsize = 100")
#     exec(f"{custom_max}.exceptions = False")

#----------- general hypers -----------#
lr = 0.0

#----------- linclass hypers -----------#
linclass_coeff = 0.0

#----------- emb hypers -----------#
emb2d_smooth_coeff = 0.0
emb3d_smooth_coeff = 0.0
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

#----------- tri2d hypers -----------#
tri_2d_smooth_coeff = 0.0
tri_2d_ml_coeff = 0.0
tri_2d_l2_coeff = 0.0
tri_2d_mindist = 0.0
tri_2d_num_samples = 0
tri_2d_ce_coeff = 0.0

#----------- col2d hypers -----------#
col2d_l1_coeff = 0.0
col2d_huber_coeff = 0.0

#----------- feat3d hypers -----------#
feat3d_sparse = False
feat3d_smooth_coeff = 0.0
feat3d_dim = 32

#----------- up3d hypers -----------#
up3d_smooth_coeff = 0.0

#----------- reject3d hypers -----------#
reject3d_ce_coeff = 0.0
reject3d_smooth_coeff = 0.0

#----------- feat2d hypers -----------#
feat2d_smooth_coeff = 0.0
feat2d_dim = 8

# --------- moc hypers ------------- #
# moc_dict_len = 50000
# num_neg_samples = 2000
# do_bn = True  # Do I have the capability of doing batch normalization
# num_pos_samples = 1000  # helpful for doing voxel level moco_learning
moc2d_coeff = 0.0
moc2d_num_samples = 1000
moc3d_coeff = 0.0
moc3d_num_samples = 1000

#----------- genocc hypers -----------#
genocc_coeff = 0.0
genocc_smooth_coeff = 0.0

#----------- gengray hypers -----------#
gengray_coeff = 0.0
gengray_smooth_coeff = 0.0

#----------- gen2dvq hypers -----------#
gen2dvq_coeff = 0.0
gen2dvq_smooth_coeff = 0.0

#----------- gen3d hypers -----------#
gen3d_coeff = 0.0
gen3d_smooth_coeff = 0.0

#----------- sigen2d hypers -----------#
sigen2d_coeff = 0.0
sigen2d_smooth_coeff = 0.0

#----------- sigen3d hypers -----------#
sigen3d_coeff = 0.0
sigen3d_smooth_coeff = 0.0

#----------- vq2d hypers -----------#
vq2d_emb_dim = 64
vq2d_num_embeddings = 64
vq2d_recon_coeff = 0.0
vq2d_latent_coeff = 0.0

#----------- vq3d hypers -----------#
# note that in vq3d, another net handles the recon
vq3d_num_embeddings = 512
vq3d_latent_coeff = 0.0

#----------- match hypers -----------#
match_coeff = 0.0
match_r_coeff = 0.0

#----------- conf hypers -----------#
conf_coeff = 0.0
# conf_num_replicas = 1

#----------- translation hypers -----------#
translation_coeff = 0.0

#----------- rigid hypers -----------#
rigid_use_cubes = False
rigid_repeats = 1
rigid_r_coeff = 0.0
rigid_t_coeff = 0.0

#----------- robust hypers -----------#
robust_corner_coeff = 0.0
robust_r_coeff = 0.0
robust_t_coeff = 0.0

#----------- obj hypers -----------#
obj_coeff = 0.0
obj_dim = 8

#----------- box hypers -----------#
box_sup_coeff = 0.0
box_cs_coeff = 0.0
box_dim = 8

#----------- ort hypers -----------#
ort_coeff = 0.0
ort_warp_coeff = 0.0
ort_dim = 8

#----------- inp hypers -----------#
inp_coeff = 0.0
inp_dim = 8

#----------- traj hypers -----------#
traj_coeff = 0.0
traj_dim = 8

#----------- preocc hypers -----------#
preocc_do_flip = False
preocc_coeff = 0.0
preocc_smooth_coeff = 0.0
preocc_reg_coeff = 0.0
preocc_density_coeff = 0.0

#----------- occ hypers -----------#
occ_coeff = 0.0
occ_smooth_coeff = 0.0

#----------- center hypers -----------#
center_focal_coeff = 0.0
center_prob_coeff = 0.0
center_size_coeff = 0.0
center_rot_coeff = 0.0

#----------- seg hypers -----------#
seg_prob_coeff = 0.0
seg_smooth_coeff = 0.0

#----------- view hypers -----------#
view_depth = 64
view_accu_render = False
view_accu_render_unps = False
view_accu_render_gt = False
view_pred_embs = False
view_pred_rgb = False
view_l1_coeff = 0.0

#----------- render hypers -----------#
render_depth = 64
render_embs = False
render_rgb = False
render_l2_coeff = 0.0

#----------- vis hypers-------------#
vis_softmax_coeff = 0.0
vis_hard_coeff = 0.0
vis_l1_coeff = 0.0
vis_debug = False

#----------- det hypers -----------#
det_anchor_size = 12.0
det_prob_coeff = 0.0
det_reg_coeff = 0.0

#----------- pri2d hypers  -----------#
pri2d_ce_coeff = 0.0
pri2d_smooth_coeff = 0.0
pri2d_reg_coeff = 0.0

#----------- pri3d hypers  -----------#
pri3d_ce_coeff = 0.0
pri3d_smooth_coeff = 0.0
pri3d_reg_coeff = 0.0

#----------- rpo2d hypers  -----------#
rpo2d_forward_coeff = 0.0
rpo2d_reverse_coeff = 0.0

#----------- rpo3d hypers  -----------#
rpo3d_forward_coeff = 0.0
rpo3d_reverse_coeff = 0.0

#----------- forecast hypers  -----------#
forecast_maxmargin_coeff = 0.0
forecast_smooth_coeff = 0.0
forecast_num_negs = 0
forecast_l2_coeff = 0.0

#----------- motioncost hypers  -----------#
motioncost_maxmargin_coeff = 0.0
motioncost_smooth_coeff = 0.0
motioncost_num_negs = 0

#----------- motionreg hypers  -----------#
motionreg_dropout = False
motionreg_t_past = 0
motionreg_t_futu = 0
motionreg_l1_coeff = 0.0
motionreg_l2_coeff = 0.0
motionreg_weak_coeff = 0.0
motionreg_vel_coeff = 0.0
motionreg_smooth_coeff = 0.0
motionreg_num_slots = 0

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

#----------- loc hypers -----------#
loc_samp_coeff = 0.0
loc_feat_coeff = 0.0

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

dataset_filetype = "tf" # can be tf or npz

# mode selection
do_zoom = False
do_carla_mot = False
do_carla_static = False
do_carla_flo = False
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
    # if do_forecast or do_match:
    #     name += "_z%dx%dx%d" % (ZZ, ZY, ZX)
    if do_forecast or do_match or do_rigid or do_robust:
        name += "_z%dx%dx%d" % (ZZ, ZY, ZX)
    
if do_view or do_emb2d or do_render:
    name += "_p%dx%d" % (PH,PW)

if pb_model:
    name += "_%s" % pb_model

if lr > 0.0:
    lrn = "%.1e" % lr
    # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1]
    name += "_%s" % lrn

if do_preocc:
    name += "_P"
    if preocc_do_flip:
        name += "l"
    if do_freeze_preocc:
        name += "f"
    preocc_coeffs = [
        preocc_coeff,
        preocc_smooth_coeff,
        preocc_reg_coeff,
        preocc_density_coeff,
    ]
    preocc_prefixes = [
        "c",
        "s",
        "r",
        "d",
    ]
    for l_, l in enumerate(preocc_coeffs):
        if l > 0:
            name += "_%s%s" % (preocc_prefixes[l_],strnum(l))
    
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
    if feat3d_sparse:
        name += "S"
    if do_freeze_feat3d:
        name += "f"
    coeffs = [
        feat3d_dim,
        feat3d_smooth_coeff,
    ]
    prefixes = [
        "d",
        "s",
    ]
    for l_, l in enumerate(coeffs):
        if l > 0:
            name += "_%s%s" % (prefixes[l_],strnum(l))

if do_up3d:
    name += "_U"
    coeffs = [
        up3d_smooth_coeff,
    ]
    prefixes = [
        "s",
    ]
    for l_, l in enumerate(coeffs):
        if l > 0:
            name += "_%s%s" % (prefixes[l_],strnum(l))

if do_moc2d:
    name += "_M2"
    coeffs = [
        moc2d_num_samples,
        moc2d_coeff,
    ]
    prefixes = [
        "s",
        "c",
    ]
    for l_, l in enumerate(coeffs):
        if l > 0:
            name += "_%s%s" % (prefixes[l_],strnum(l))
if do_moc3d:
    name += "_M3"
    coeffs = [
        moc3d_num_samples,
        moc3d_coeff,
    ]
    prefixes = [
        "s",
        "c",
    ]
    for l_, l in enumerate(coeffs):
        if l > 0:
            name += "_%s%s" % (prefixes[l_],strnum(l))

if do_vq2d:
    name += "_Vr"
    if do_freeze_vq2d:
        name += "f"
    vq2d_coeffs = [
        vq2d_num_embeddings,
        vq2d_emb_dim,
        vq2d_recon_coeff,
        vq2d_latent_coeff,
    ]
    vq2d_prefixes = [
        "n",
        "d",
        "r",
        "l",
    ]
    for l_, l in enumerate(vq2d_coeffs):
        if l > 0:
            name += "_%s%s" % (vq2d_prefixes[l_],strnum(l))

if do_vq3d:
    name += "_V3r"
    vq3d_coeffs = [
        vq3d_num_embeddings,
        vq3d_latent_coeff,
    ]
    if do_freeze_vq3d:
        name += "f"
    vq3d_prefixes = [
        "n",
        "l",
    ]
    for l_, l in enumerate(vq3d_coeffs):
        if l > 0:
            name += "_%s%s" % (vq3d_prefixes[l_],strnum(l))

if do_genocc:
    name += "_Go"
    genocc_losses = [
        genocc_coeff,
        genocc_smooth_coeff,
    ]
    genocc_prefixes = [
        "c",
        "s",
    ]
    for l_, l in enumerate(genocc_losses):
        if l > 0:
            name += "_%s%s" % (genocc_prefixes[l_],strnum(l))
            
if do_gengray:
    name += "_Gg"
    gengray_losses = [
        gengray_coeff,
        gengray_smooth_coeff,
    ]
    gengray_prefixes = [
        "c",
        "s",
    ]
    for l_, l in enumerate(gengray_losses):
        if l > 0:
            name += "_%s%s" % (gengray_prefixes[l_],strnum(l))

if do_gen2dvq:
    name += "_G2v"
    gen2dvq_losses = [
        gen2dvq_coeff,
        gen2dvq_smooth_coeff,
    ]
    gen2dvq_prefixes = [
        "c",
        "s",
    ]
    for l_, l in enumerate(gen2dvq_losses):
        if l > 0:
            name += "_%s%s" % (gen2dvq_prefixes[l_],strnum(l))

if do_gen3d:
    name += "_G3v"
    if do_freeze_gen3d:
        name += "f"
    gen3d_losses = [
        gen3d_coeff,
        gen3d_smooth_coeff,
    ]
    gen3d_prefixes = [
        "c",
        "s",
    ]
    for l_, l in enumerate(gen3d_losses):
        if l > 0:
            name += "_%s%s" % (gen3d_prefixes[l_],strnum(l))

if do_sigen2d:
    name += "_S2i"
    if do_freeze_sigen2d:
        name += "f"
    sigen2d_losses = [
        sigen2d_coeff,
        sigen2d_smooth_coeff,
    ]
    sigen2d_prefixes = [
        "c",
        "s",
    ]
    for l_, l in enumerate(sigen2d_losses):
        if l > 0:
            name += "_%s%s" % (sigen2d_prefixes[l_],strnum(l))

if do_sigen3d:
    name += "_S3i"
    if do_freeze_sigen3d:
        name += "f"
    sigen3d_losses = [
        sigen3d_coeff,
        sigen3d_smooth_coeff,
    ]
    sigen3d_prefixes = [
        "c",
        "s",
    ]
    for l_, l in enumerate(sigen3d_losses):
        if l > 0:
            name += "_%s%s" % (sigen3d_prefixes[l_],strnum(l))

if do_match:
    name += "_M"
    if do_freeze_match:
        name += "f"
    match_losses = [
        match_coeff,
        match_r_coeff,
    ]
    match_prefixes = [
        "c",
        "r",
    ]
    for l_, l in enumerate(match_losses):
        if l > 0:
            name += "_%s%s" % (match_prefixes[l_],strnum(l))

if do_conf:
    name += "_C"
    if do_freeze_conf:
        name += "f"
    conf_coeffs = [
        conf_coeff,
        # conf_num_replicas,
    ]
    conf_prefixes = [
        "c",
        # "r",
    ]
    for l_, l in enumerate(conf_coeffs):
        if l > 0:
            name += "_%s%s" % (conf_prefixes[l_],strnum(l))

if do_translation:
    name += "_T"
    translation_losses = [
        translation_coeff,
    ]
    translation_prefixes = [
        "c",
    ]
    for l_, l in enumerate(translation_losses):
        if l > 0:
            name += "_%s%s" % (translation_prefixes[l_],strnum(l))

if do_rigid:
    name += "_Ri"
    if rigid_use_cubes:
        name += "c"
    if rigid_repeats > 1:
        name += "%d" % rigid_repeats
    rigid_losses = [
        rigid_r_coeff,
        rigid_t_coeff,
    ]
    rigid_prefixes = [
        "r",
        "t",
    ]
    for l_, l in enumerate(rigid_losses):
        if l > 0:
            name += "_%s%s" % (rigid_prefixes[l_],strnum(l))

if do_robust:
    name += "_Ro"
    robust_losses = [
        robust_corner_coeff,
        robust_r_coeff,
        robust_t_coeff,
    ]
    robust_prefixes = [
        "c",
        "r",
        "t",
    ]
    for l_, l in enumerate(robust_losses):
        if l > 0:
            name += "_%s%s" % (robust_prefixes[l_],strnum(l))

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
            ego_losses = [ego_rtd_coeff,
                          ego_rta_coeff,
                          ego_traj_coeff,
                          ego_warp_coeff,
            ]
            ego_prefixes = ["rtd",
                            "rta",
                            "t",
                            "w",
            ]
            for l_, l in enumerate(ego_losses):
                if l > 0:
                    name += "_%s%s" % (ego_prefixes[l_],strnum(l))

if do_obj:
    name += "_J"
    # name += "%d" % obj_dim

    if do_freeze_obj:
        name += "f"
    else:
        # no real hyps here
        pass

if do_box:
    name += "_B"
    # name += "%d" % box_dim

    if do_freeze_box:
        name += "f"
    else:
        box_coeffs = [box_sup_coeff,
                      box_cs_coeff,
                      # box_smooth_coeff,
        ]
        box_prefixes = ["su",
                        "cs",
                        # "s",
        ]
        for l_, l in enumerate(box_coeffs):
            if l > 0:
                name += "_%s%s" % (box_prefixes[l_],strnum(l))


if do_ort:
    name += "_O"
    # name += "%d" % ort_dim

    if do_freeze_ort:
        name += "f"
    else:
        ort_coeffs = [ort_coeff,
                      ort_warp_coeff,
                      # ort_smooth_coeff,
        ]
        ort_prefixes = ["c",
                        "w",
                        # "s",
        ]
        for l_, l in enumerate(ort_coeffs):
            if l > 0:
                name += "_%s%s" % (ort_prefixes[l_],strnum(l))

if do_inp:
    name += "_I"
    # name += "%d" % inp_dim

    if do_freeze_inp:
        name += "f"
    else:
        inp_coeffs = [inp_coeff,
                      # inp_smooth_coeff,
        ]
        inp_prefixes = ["c",
                        # "s",
        ]
        for l_, l in enumerate(inp_coeffs):
            if l > 0:
                name += "_%s%s" % (inp_prefixes[l_],strnum(l))

if do_traj:
    name += "_T"
    name += "%d" % traj_dim

    if do_freeze_traj:
        name += "f"
    else:
        # no real hyps here
        pass

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

if do_center:
    name += "_C"
    if do_freeze_center:
        name += "f"
    center_coeffs = [
        center_focal_coeff,
        center_prob_coeff,
        center_size_coeff,
        center_rot_coeff,
    ]
    center_prefixes = [
        "f",
        "p",
        "s",
        "r",
    ]
    for l_, l in enumerate(center_coeffs):
        if l > 0:
            name += "_%s%s" % (center_prefixes[l_],strnum(l))

if do_seg:
    name += "_S"
    if do_freeze_seg:
        name += "f"
    seg_coeffs = [
        seg_prob_coeff,
        seg_smooth_coeff,
    ]
    seg_prefixes = [
        "p",
        "s",
    ]
    for l_, l in enumerate(seg_coeffs):
        if l > 0:
            name += "_%s%s" % (seg_prefixes[l_],strnum(l))

if do_reject3d:
    name += "_F3"
    if do_freeze_reject3d:
        name += "f"
    coeffs = [
        reject3d_ce_coeff,
        reject3d_smooth_coeff,
    ]
    prefixes = [
        "c",
        "s",
    ]
    for l_, l in enumerate(coeffs):
        if l > 0:
            name += "_%s%s" % (prefixes[l_],strnum(l))

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

if do_render:
    name += "_R"
    if render_embs:
        name += "e"
    if render_rgb:
        name += "r"
    if do_freeze_render:
        name += "f"

    render_coeffs = [
        render_depth,
        render_l2_coeff,
    ]
    render_prefixes = [
        "d",
        "c",
    ]
    for l_, l in enumerate(render_coeffs):
        if l > 0:
            name += "_%s%s" % (render_prefixes[l_],strnum(l))

if do_vis:
    name += "_V"
    if vis_debug:
        name += 'd'
    if do_freeze_vis:
        name += "f"
    else:
        vis_coeffs = [vis_softmax_coeff,
                      vis_hard_coeff,
                      vis_l1_coeff,
        ]
        vis_prefixes = ["s",
                        "h",
                        "c",
        ]
        for l_, l in enumerate(vis_coeffs):
            if l > 0:
                name += "_%s%s" % (vis_prefixes[l_],strnum(l))

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
            
if do_pri2d:
    name += "_Pri2d"
    if do_freeze_pri2d:
        name += "f"
    pri_losses = [
        pri2d_ce_coeff,
        pri2d_smooth_coeff,
        pri2d_reg_coeff,
    ]
    pri_prefixes = [
        "c",
        "s",
        "r",
    ]
    for l_, l in enumerate(pri_losses):
        if l > 0:
            name += "_%s%s" % (pri_prefixes[l_],strnum(l))
if do_pri3d:
    name += "_Pri3d"
    if do_freeze_pri3d:
        name += "f"
    pri_losses = [
        pri3d_ce_coeff,
        pri3d_smooth_coeff,
        pri3d_reg_coeff,
    ]
    pri_prefixes = [
        "c",
        "s",
        "r",
    ]
    for l_, l in enumerate(pri_losses):
        if l > 0:
            name += "_%s%s" % (pri_prefixes[l_],strnum(l))
if do_rpo2d:
    name += "_Rpo2d"
    rpo2d_losses = [
        rpo2d_forward_coeff,
        rpo2d_reverse_coeff,
    ]
    rpo2d_prefixes = [
        "f",
        "r",
    ]
    for l_, l in enumerate(rpo2d_losses):
        if l > 0:
            name += "_%s%s" % (rpo2d_prefixes[l_],strnum(l))
if do_rpo3d:
    name += "_Rpo3d"
    rpo3d_losses = [
        rpo3d_forward_coeff,
        rpo3d_reverse_coeff,
    ]
    rpo3d_prefixes = [
        "f",
        "r",
    ]
    for l_, l in enumerate(rpo3d_losses):
        if l > 0:
            name += "_%s%s" % (rpo3d_prefixes[l_],strnum(l))
if do_forecast:
    name += "_Fo"
    if do_freeze_forecast:
        name += "f"
    forecast_losses = [
        forecast_num_negs,
        forecast_maxmargin_coeff,
        forecast_smooth_coeff,
        forecast_l2_coeff,
    ]
    forecast_prefixes = [
        "n",
        "m",
        "s",
        "e",
    ]
    for l_, l in enumerate(forecast_losses):
        if l > 0:
            name += "_%s%s" % (forecast_prefixes[l_],strnum(l))
if do_motioncost:
    name += "_Mo"
    if do_freeze_motioncost:
        name += "f"
    motioncost_coeffs = [
        motioncost_num_negs,
        motioncost_maxmargin_coeff,
        motioncost_smooth_coeff,
    ]
    motioncost_prefixes = [
        "n",
        "m",
        "s",
    ]
    for l_, l in enumerate(motioncost_coeffs):
        if l > 0:
            name += "_%s%s" % (motioncost_prefixes[l_],strnum(l))
if do_motionreg:
    name += "_Mr"
    if do_freeze_motionreg:
        name += "f"
    if motionreg_dropout:
        name += "d"
    motionreg_coeffs = [
        motionreg_t_past,
        motionreg_t_futu,
        motionreg_num_slots,
        motionreg_l1_coeff,
        motionreg_l2_coeff,
        motionreg_weak_coeff,
        motionreg_smooth_coeff,
        motionreg_vel_coeff,
    ]
    motionreg_prefixes = [
        "p",
        "f",
        "k",
        "c",
        "e",
        "w",
        "s",
        "v",
    ]
    for l_, l in enumerate(motionreg_coeffs):
        if l > 0:
            name += "_%s%s" % (motionreg_prefixes[l_],strnum(l))
            
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
if do_tri2d:
    name += "_T2"
    coeffs = [
        tri_2d_ml_coeff,
        tri_2d_l2_coeff,
        tri_2d_num_samples,
        tri_2d_mindist,
        tri_2d_ce_coeff,
    ]
    prefixes = [
        "s",
        "m",
        "e",
        "n",
        "d",
        "c",
    ]
    for l_, l in enumerate(coeffs):
        if l > 0:
            name += "_%s%s" % (prefixes[l_],strnum(l))

if do_col2d:
    name += "_C2"
    coeffs = [
        col2d_l1_coeff,
        col2d_huber_coeff,
    ]
    prefixes = [
        "c",
        "h",
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

if do_loc:
    name += "_L"
    if do_freeze_loc:
        name += "l"
    coeffs = [
        loc_samp_coeff,
        loc_feat_coeff,
    ]
    prefixes = [
        "c",
        "f",
    ]
    for l_, l in enumerate(coeffs):
        if l > 0:
            name += "_%s%s" % (prefixes[l_],strnum(l))

if do_linclass:
    name += "_L"
    if do_freeze_linclass:
        name += "f"
    coeffs = [
        linclass_coeff,
    ]
    prefixes = [
        "c",
    ]
    for l_, l in enumerate(coeffs):
        if l > 0:
            name += "_%s%s" % (prefixes[l_],strnum(l))
                

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
