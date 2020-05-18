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
                 "viewnet": hyp.view_init,
                 "detnet": hyp.det_init,
                 "flownet": hyp.flow_init,
                 "egonet": hyp.ego_init,
                 "occnet": hyp.occ_init,
        }

        for part, init in list(inits.items()):
            # print('part', part)
            if init:
                if part == 'feat2dnet':
                    model_part = model.feat2dnet
                elif part == 'feat3dnet':
                    model_part = model.feat3dnet
                elif part == 'occnet':
                    model_part = model.occnet
                elif part == 'flownet':
                    model_part = model.flownet
                else:
                    assert(False)
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
            model_name = ckpt_names[ind]
            path = os.path.join(init_dir, model_name)
            print("...found checkpoint %s" % (path))

            checkpoint = torch.load(path)
            model_state_dict = model.state_dict()
            # print(model_state_dict.keys())
            for load_param_name, param in checkpoint['model_state_dict'].items():
                model_param_name = load_param_name[len(part)+1:]
                # print('load_param_name', load_param_name, 'model_param_name', model_param_name)
                if part+"."+model_param_name != load_param_name:
                    continue
                else:
                    if model_param_name in model_state_dict.keys():
                        # print(model_param_name, load_param_name)
                        # print('param in ckpt', param.data.shape)
                        # print('param in state dict', model_state_dict[model_param_name].shape)
                        model_state_dict[model_param_name].copy_(param.data)
                    else:
                        print('warning: %s is not in the state dict of the current model' % model_param_name)
        else:
            print("...ain't no %s checkpoint here!"%(part))
    return step

