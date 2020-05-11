import torch
import os,pathlib
import hyperparams as hyp
import numpy as np


def load_total(model, optimizer):
    start_iter = 0
    if hyp.total_init:
        print("TOTAL INIT")
        print(hyp.total_init)
        start_iter = load(hyp.total_init, model, optimizer)
        if start_iter:
            print("loaded full model. resuming from iter %08d" % start_iter)
        else:
            print("could not find a full model. starting from scratch")
    return start_iter

def load_weights(model, optimizer):
    if hyp.total_init:
        print("TOTAL INIT")
        print(hyp.total_init)
        start_iter = load(hyp.total_init, model, optimizer)
        if start_iter:
            print("loaded full model. resuming from iter %08d" % start_iter)
        else:
            print("could not find a full model. starting from scratch")
    else:
        # if (1):
        start_iter = 0
        inits = {"feat2dnet": hyp.feat2d_init,
                 "feat3dnet": hyp.feat3d_init,
                 "up3dnet": hyp.up3d_init,
                 "segnet": hyp.seg_init,
                 "viewnet": hyp.view_init,
                 "segnet": hyp.seg_init,
                 "matchnet": hyp.match_init,
                 "confnet": hyp.conf_init,
                 "rigidnet": hyp.rigid_init,
                 "locnet": hyp.loc_init,
                 "vq2dnet": hyp.vq2d_init,
                 "vq3dnet": hyp.vq3d_init,
                 "sigen3dnet": hyp.sigen3d_init,
                 "motionregnet": hyp.motionreg_init,
                 "gen3dnet": hyp.gen3d_init,
                 "forecastnet": hyp.forecast_init,
                 "detnet": hyp.det_init,
                 "visnet": hyp.vis_init,
                 "flownet": hyp.flow_init,
                 "pri2dnet": hyp.pri2d_init,
                 "emb2dnet": hyp.emb2d_init,
                 # "emb3dnet": hyp.emb3d_init, # no params here really
                 "inpnet": hyp.inp_init,
                 "egonet": hyp.ego_init,
                 "occnet": hyp.occ_init,
                 "centernet": hyp.center_init,
                 "preoccnet": hyp.preocc_init,
                 "optim": hyp.optim_init,
                 # "obj": hyp.obj_init,
                 # "bkg": hyp.bkg_init,
        }

        for part, init in list(inits.items()):
            # print('part', part)
            if init:
                if part == 'feat2dnet':
                    model_part = model.feat2dnet
                elif part == 'feat3dnet':
                    model_part = model.feat3dnet
                # elif part == 'up3dnet':
                #     model_part = model.up3dnet
                # elif part == 'viewnet':
                #     model_part = model.viewnet
                # elif part == 'segnet':
                #     model_part = model.segnet
                # elif part == 'matchnet':
                #     model_part = model.matchnet
                # elif part == 'confnet':
                #     model_part = model.confnet
                # elif part == 'rigidnet':
                #     model_part = model.rigidnet
                # elif part == 'locnet':
                #     model_part = model.locnet
                # elif part == 'vq2dnet':
                #     model_part = model.vq2dnet
                # elif part == 'vq3dnet':
                #     model_part = model.vq3dnet
                # elif part == 'sigen3dnet':
                #     model_part = model.sigen3dnet
                # elif part == 'motionregnet':
                #     model_part = model.motionregnet
                # elif part == 'gen3dnet':
                #     model_part = model.gen3dnet
                # elif part == 'forecastnet':
                #     model_part = model.forecastnet
                # elif part == 'detnet':
                #     model_part = model.detnet
                # elif part == 'occnet':
                #     model_part = model.occnet
                # elif part == 'centernet':
                #     model_part = model.centernet
                # elif part == 'preoccnet':
                #     model_part = model.preoccnet
                # elif part == 'pri2dnet':
                #     model_part = model.pri2dnet
                elif part == 'emb2dnet':
                    model_part = model.emb2dnet
                elif part == 'flownet':
                    model_part = model.flownet
                elif part == 'optim':
                    model_part = model.optim_dict
                # elif part == 'bkg':
                #     model_part = model.bkg
                # elif part == 'obj':
                #     model_part = [
                #         model.obj,
                #         model.obj_alist,
                #         model.obj_tlist,
                #         model.obj_l,
                #     ]
                else:
                    assert(False)
                if part=='optim':
                    load_optim(model_part, init)
                else:
                    if isinstance(model_part, list):
                        for mp in model_part:
                            iter = load_part([mp], part, init)
                    else:
                        iter = load_part(model_part, part, init)
                if iter:
                    print("loaded %s at iter %08d" % (init, iter))
                else:
                    print("could not find a checkpoint for %s" % init)
    if hyp.reset_iter:
        start_iter = 0
    return start_iter


def save(model, checkpoint_dir, step, optimizer, keep_latest=3):
    model_name = "model-%08d.pth"%(step)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    prev_chkpts = list(pathlib.Path(checkpoint_dir).glob('model-*'))
    prev_chkpts.sort(key=lambda p: p.stat().st_mtime,reverse=True)
    if len(prev_chkpts) > keep_latest-1:
        for f in prev_chkpts[keep_latest-1:]:
            f.unlink()
    path = os.path.join(checkpoint_dir, model_name)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, path)
    print("Saved a checkpoint: %s"%(path))

def save_optim(model, checkpoint_dir, step):
    print('SAVE_OPTIM: NOT YET SAVING ANY MOMENTUM PARAMS')
    model_name = "model-%08d-optim.npz" % (step)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    path = os.path.join(checkpoint_dir, model_name)
    save_dict = {}
    for k, v in model.optim_dict.items():
        print('saving %s' % k)
        save_dict[k] = v.detach().cpu().numpy()
    np.savez(path, save_dict=save_dict)
    print("Saved an optim checkpoint: %s" % (path))
    
def load_optim(model_optim_dict, load_dir):
    checkpoint_dir = os.path.join("checkpoints/", load_dir)
    print('LOAD_OPTIM: NOT YET LOADING ANY MOMENTUM PARAMS')
    if not os.path.exists(checkpoint_dir):
        print("...ain't no full checkpoint here!")
    else:
        ckpt_names = os.listdir(checkpoint_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            # ind, step = max(steps)
            # model_name = "model-%08d-optim.npz" % (step)
            ind = np.argmax(steps)
            # model_name = "model-%08d-optim.npz" % (step)
            model_name = ckpt_names[ind]
            path = os.path.join(checkpoint_dir, model_name)
            print('loading from %s' % path)
            load_dict = np.load(path, allow_pickle=True)['save_dict']
            load_dict = load_dict.item()
            # print(load_dict.files)
            # print(o)
            # o = o.item()
            # print(o)
            # load_dict = o.files['save_dict']
            # print(load_dict)
            for k, v in model_optim_dict.items():
                print('loading %s' % k)
                v.data = torch.FloatTensor(load_dict[k]).to(torch.device('cuda'))
            print('done loading')
        else:
            print("...ain't no full checkpoint here!")

def load(model_name, model, optimizer):
    print("reading full checkpoint...")
    # checkpoint_dir = os.path.join("checkpoints/", model_name)
    checkpoint_dir = os.path.join("saved_checkpoints/", model_name)
    step = 0
    if not os.path.exists(checkpoint_dir):
        print("...ain't no full checkpoint here!")
    else:
        ckpt_names = os.listdir(checkpoint_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            step = max(steps)
            model_name = 'model-%08d.pth' % (step)
            path = os.path.join(checkpoint_dir, model_name)
            print("...found checkpoint %s"%(path))

            checkpoint = torch.load(path)
            
            # # Print model's state_dict
            # print("Model's state_dict:")
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            # input()

            # # Print optimizer's state_dict
            # print("Optimizer's state_dict:")
            # for var_name in optimizer.state_dict():
            #     print(var_name, "\t", optimizer.state_dict()[var_name])
            # input()
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("...ain't no full checkpoint here!")
    return step

def load_part(model, part, init):
    print("reading %s checkpoint..." % part)
    init_dir = os.path.join("saved_checkpoints", init)
    print(init_dir)
    step = 0
    if not os.path.exists(init_dir):
        print("...ain't no %s checkpoint here!"%(part))
    else:
        ckpt_names = os.listdir(init_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            step = max(steps)
            ind = np.argmax(steps)
            # model_name = 'model-%08d.pth'%(step)
            model_name = ckpt_names[ind]
            path = os.path.join(init_dir, model_name)
            print("...found checkpoint %s"%(path))

            checkpoint = torch.load(path)
            model_state_dict = model.state_dict()
            # print(model_state_dict.keys())
            for load_para_name, para in checkpoint['model_state_dict'].items():
                model_para_name = load_para_name[len(part)+1:]
                # print(model_para_name, load_para_name)
                if part+"."+model_para_name != load_para_name:
                    continue
                else:
                    if model_para_name in model_state_dict.keys():
                        print(model_para_name, load_para_name)
                        print('param in ckpt', para.data.shape)
                        print('param in state dict', model_state_dict[model_para_name].shape)
                        model_state_dict[model_para_name].copy_(para.data)
                    else:
                        print('warning: %s is not in the state dict of the current model' % model_para_name)
        else:
            print("...ain't no %s checkpoint here!"%(part))
    return step

