import sys
import os
import torch
import copy

checkpoint = sys.argv[1]
target_mp = int(sys.argv[2])

assert os.path.isdir(checkpoint)
with open(os.path.join(checkpoint, 'latest_checkpointed_iteration.txt')) as fin:
    iteration = int(fin.read().strip())

checkpoint = os.path.join(checkpoint, str(iteration))

filenames = os.listdir(checkpoint)
filenames = sorted(filenames, 
        key=lambda x: int(x.split('_')[2]))
filenames = [os.path.join(checkpoint, x) for x in filenames]

if target_mp == len(filenames):
    print("MP size keeps the same.")
    exit(0)

if sys.argv[1][-1] == '/':
    new_checkpoint = sys.argv[1][:-1] + '_MP' + sys.argv[2]
else:
    new_checkpoint = sys.argv[1] + '_MP' + sys.argv[2]
if not os.path.exists(new_checkpoint):
    os.mkdir(new_checkpoint)
with open(os.path.join(new_checkpoint, 'latest_checkpointed_iteration.txt'), 'w') as fout:
    fout.write("{}\n".format(iteration))
new_checkpoint = os.path.join(new_checkpoint, str(iteration))
if not os.path.exists(new_checkpoint):
    os.mkdir(new_checkpoint)

preserve_keys = [
    "lr_scheduler",
    "skipped_steps",
    "global_steps",
    "global_samples",
    "dp_world_size",
    "iteration",
    "np_rng_state",
    "random_rng_state",
    "torch_rng_state",
    "cuda_rng_state",
    "rng_tracker_states",
    
]

if target_mp < len(filenames):
    print("Decrease MP size.")
    assert len(filenames) % target_mp == 0
    ratio = len(filenames) // target_mp
    for i in range(target_mp):
        start = ratio * i
        end = ratio * (i+1)
        d = torch.load(filenames[start], 
                map_location='cpu')
        for k in d.keys():
            if k !='module':
                if k in preserve_keys:
                    pass
                elif k == "mp_world_size":
                    d[k] = target_mp
                else:
                    d[k] = None
        for j in range(start+1, end):
            d_new = torch.load(filenames[j], 
                    map_location='cpu')
            for k, v in d_new['module'].items():
                assert len(v.shape) < 3
                if len(v.shape) == 2 and 'position' not in k:
                    if 'query' in k:
                        size_1 = d['module'][k].shape[0] // 3
                        size_2 = v.shape[0] // 3
                        target = d['module'][k]
                        d['module'][k] = torch.cat([
                            target[:size_1, :], v[:size_2, :],
                            target[size_1:size_1*2, :], v[size_2:size_2*2, :],
                            target[size_1*2:, :], v[size_2*2:, :]], 0)
                    elif 'word' in k  or 'h_to_4h' in k:
                        d['module'][k] = torch.cat([d['module'][k], v], 0)
                    else:
                        d['module'][k] = torch.cat([d['module'][k], v], 1)
                if len(v.shape) == 1 and 'query_key_value' in k:
                    size_1 = d['module'][k].shape[0] // 3
                    size_2 = v.shape[0] // 3
                    target = d['module'][k]
                    d['module'][k] = torch.cat([
                        target[:size_1], v[:size_2],
                        target[size_1:size_1*2], v[size_2:size_2*2],
                        target[size_1*2:], v[size_2*2:]], 0)

                if len(v.shape) == 1 and 'dense_h_to_4h' in k:
                    d['module'][k] = torch.cat([d['module'][k], v], 0)
        filename = os.path.join(new_checkpoint, "mp_rank_{:02d}_model_states.pt".format(i))
        torch.save(d, filename)

if target_mp > len(filenames):
    print("Increase MP size.")
    assert target_mp % len(filenames) == 0
    ratio = target_mp // len(filenames)
    for i in range(len(filenames)):
        start = ratio * i
        end = ratio * (i+1)
        d = torch.load(filenames[i], 
                map_location='cpu')
        for j in range(start, end):
            d_new = {}
            shift = j - start
            for k, v in d.items():
                if k != 'module':
                    if k in preserve_keys:
                        d_new[k] = copy.deepcopy(d[k])
                    elif k == "mp_world_size":
                        d_new[k] = target_mp
                    else:
                        d_new[k] = None
            d_new['module'] = {}
            for k, v in d['module'].items():
                assert len(v.shape) < 3
                if len(v.shape) == 2 and 'position' not in k:
                    if 'query' in k:
                        part = v.shape[0] // ratio // 3
                        d_new['module'][k] = torch.cat([v[shift*part:(shift+1)*part, :], v[(shift+ratio)*part:(shift+1+ratio)*part, :], v[(shift+2*ratio)*part:(shift+1+2*ratio)*part, :]], 0)
                    elif 'word' in k or 'h_to_4h' in k:
                        part = v.shape[0] // ratio
                        d_new['module'][k] = v[shift*part:(shift+1)*part, :]
                    else:
                        part = v.shape[1] // ratio
                        d_new['module'][k] = v[:, shift*part:(shift+1)*part]
                elif len(v.shape) == 1 and 'dense_h_to_4h' in k:
                    part = v.shape[0] // ratio
                    d_new['module'][k] = v[shift*part:(shift+1)*part]
                elif len(v.shape) == 1 and 'query_key_value' in k:
                    part = v.shape[0] // ratio // 3
                    d_new['module'][k] = torch.cat([v[shift*part:(shift+1)*part], v[(shift+ratio)*part:(shift+1+ratio)*part], v[(shift+2*ratio)*part:(shift+1+2*ratio)*part]], 0)
                else:
                    d_new['module'][k] = v
            filename = os.path.join(new_checkpoint, "mp_rank_{:02d}_model_states.pt".format(j))
            torch.save(d_new, filename)
