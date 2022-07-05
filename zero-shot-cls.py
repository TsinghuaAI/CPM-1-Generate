import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from arguments import get_args
from data_utils.tokenization_gpt2 import GPT2Tokenizer
import mpu
import json

from data.samplers import DistributedBatchSampler, RandomSampler

from torch.utils.data import TensorDataset

from generate_samples import *

def get_batch(context_tokens, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        args.reset_position_ids,
        args.reset_attention_mask)

    return tokens, attention_mask, position_ids

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        #convert to 1D
        logits=logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        #going back to 2D
        logits=logits.view(1, -1).contiguous()
	
    return logits



def prepare_tokenizer(args):

    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir}
    tokenizer = make_tokenizer(**tokenizer_args)

    args.tokenizer_num_tokens = tokenizer.num_tokens
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    after = tokenizer.num_tokens
    while after % mpu.get_model_parallel_world_size() != 0:
        after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer

def load_ocnli_data(data_path, data_type, tokenizer):
    args = get_args()

    filename = os.path.join(data_path, data_type+'.json')
    objs = []
    with open(filename) as fin:
        for line in fin:
            objs.append(json.loads(line.strip()))

    pad_id = tokenizer.encoder['<pad>']
    args.eod_token = tokenizer.encoder['<eod>']

    all_tokens_1 = []
    all_masks_1 = []
    all_tokens_2 = []
    all_masks_2 = []    
    all_tokens_3 = []
    all_masks_3 = [] 
    all_labels = []
    for obj in objs:

        if obj['label'] == '-':
            continue

        prompt = "{}？对，".format(obj['sentence1'])
        prompt_tokens = tokenizer.encode(prompt)
        prompt_len = len(prompt_tokens)
        tokens = prompt_tokens + tokenizer.encode(obj['sentence2'])
        second_mask = [0] * (args.seq_length-1)
        for idx in range(prompt_len-1, len(tokens)-1):
            second_mask[idx] = 1
        all_masks_1.append(second_mask)
        token_length = len(tokens)
        assert token_length < args.seq_length
        tokens.extend([pad_id] * (args.seq_length - token_length))
        all_tokens_1.append(tokens)

        prompt = "{}？错，".format(obj['sentence1'])
        prompt_tokens = tokenizer.encode(prompt)
        prompt_len = len(prompt_tokens)
        tokens = prompt_tokens + tokenizer.encode(obj['sentence2'])
        second_mask = [0] * (args.seq_length-1)
        for idx in range(prompt_len-1, len(tokens)-1):
            second_mask[idx] = 1
        all_masks_2.append(second_mask)
        token_length = len(tokens)
        assert token_length < args.seq_length
        tokens.extend([pad_id] * (args.seq_length - token_length))
        all_tokens_2.append(tokens)

        prompt = "{}？也许，".format(obj['sentence1'])
        prompt_tokens = tokenizer.encode(prompt)
        prompt_len = len(prompt_tokens)
        tokens = prompt_tokens + tokenizer.encode(obj['sentence2'])
        second_mask = [0] * (args.seq_length-1)
        for idx in range(prompt_len-1, len(tokens)-1):
            second_mask[idx] = 1
        all_masks_3.append(second_mask)
        token_length = len(tokens)
        assert token_length < args.seq_length
        tokens.extend([pad_id] * (args.seq_length - token_length))
        all_tokens_3.append(tokens)

        if obj['label'] == 'entailment':
            all_labels.append([0])
        elif obj['label'] == 'contradiction':
            all_labels.append([1])
        else:
            all_labels.append([2])

    all_tokens_1 = torch.tensor(all_tokens_1, dtype=torch.long)
    all_masks_1 = torch.tensor(all_masks_1, dtype=torch.float)
    all_tokens_2 = torch.tensor(all_tokens_2, dtype=torch.long)
    all_masks_2 = torch.tensor(all_masks_2, dtype=torch.float)
    all_tokens_3 = torch.tensor(all_tokens_3, dtype=torch.long)
    all_masks_3 = torch.tensor(all_masks_3, dtype=torch.float)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    dataset = TensorDataset(all_tokens_1, all_masks_1, all_tokens_2, all_masks_2, all_tokens_3, all_masks_3, all_labels)

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    # Use a random sampler with distributed batch sampler.
    if data_type == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)
    
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)

def load_iflytek_data(data_path, data_type, tokenizer, sampled_labels=False):
    args = get_args()

    filename = os.path.join(data_path, data_type+'.json')
    objs = []
    with open(filename) as fin:
        for line in fin:
            objs.append(json.loads(line.strip()))

    pad_id = tokenizer.encoder['<pad>']
    args.eod_token = tokenizer.encoder['<eod>']

    labels = []
    label_map = {}
    with open(os.path.join(data_path, 'labels.json')) as fin:
        for i, line in enumerate(fin):
            obj = json.loads(line.strip())
            labels.append(obj['label_des'])
            label_map[obj['label_des']] = i

    all_tokens = []
    all_masks = []
    all_labels = []
    for _, obj in enumerate(objs):
        sentence = obj['sentence']
        tokenized_sentence = tokenizer.encode(sentence)[:args.seq_length-20]

        if sampled_labels:
            cur_labels = random.sample(labels, 3)
            while obj['label_des'] in cur_labels:
                cur_labels = random.sample(labels, 3)
            cur_labels.append(obj['label_des'])
            cur_label = cur_labels.index(obj['label_des'])
            assert cur_label != -1
        else:
            cur_labels = labels
            cur_label = label_map[obj['label_des']]
        
        all_labels.append(cur_label)

        for _, label in enumerate(cur_labels):
            prompt = "这是关于{}的应用程序：".format(label)
            prompt_tokens = tokenizer.encode(prompt)
            prompt_len = len(prompt_tokens)
            tokens = prompt_tokens + tokenized_sentence
            second_mask = [0] * (args.seq_length-1)
            for idx in range(prompt_len-1, len(tokens)-1):
                second_mask[idx] = 1
            all_masks.append(second_mask)
            token_length = len(tokens)
            assert token_length < args.seq_length
            tokens.extend([pad_id] * (args.seq_length - token_length))
            all_tokens.append(tokens)
    
    all_tokens = torch.tensor(all_tokens, dtype=torch.long)
    all_masks = torch.tensor(all_masks, dtype=torch.float)
    dataset = TensorDataset(all_tokens, all_masks)

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)
    
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True), all_labels

def load_tnews_data(data_path, data_type, tokenizer, sampled_labels=False):
    args = get_args()

    filename = os.path.join(data_path, data_type+'.json')
    objs = []
    with open(filename) as fin:
        for line in fin:
            objs.append(json.loads(line.strip()))

    pad_id = tokenizer.encoder['<pad>']
    args.eod_token = tokenizer.encoder['<eod>']

    labels = []
    label_map = {}
    label_reverse = {}
    with open(os.path.join(data_path, 'labels.json')) as fin:
        for i, line in enumerate(fin):
            obj = json.loads(line.strip())
            labels.append(obj['label_desc'])
            label_map[obj['label_desc']] = i
            label_reverse[obj['label']] = obj['label_desc']

    all_tokens = []
    all_masks = []
    all_labels = []
    for _, obj in enumerate(objs):
        sentence = obj['sentence']
        tokenized_sentence = tokenizer.encode(sentence)[:args.seq_length-20]
        obj['label_desc'] = label_reverse[obj['label']]

        if sampled_labels:
            cur_labels = random.sample(labels, 3)
            while obj['label_desc'] in cur_labels:
                cur_labels = random.sample(labels, 3)
            cur_labels.append(obj['label_desc'])
            cur_label = cur_labels.index(obj['label_desc'])
            assert cur_label != -1
        else:
            cur_labels = labels
            cur_label = label_map[obj['label_desc']]

        all_labels.append(cur_label)

        for _, label in enumerate(cur_labels):
            prompt = "这是关于{}的文章：".format(label)
            prompt_tokens = tokenizer.encode(prompt)
            prompt_len = len(prompt_tokens)
            tokens = prompt_tokens + tokenized_sentence
            second_mask = [0] * (args.seq_length-1)
            for idx in range(prompt_len-1, len(tokens)-1):
                second_mask[idx] = 1
            all_masks.append(second_mask)
            token_length = len(tokens)
            assert token_length < args.seq_length
            tokens.extend([pad_id] * (args.seq_length - token_length))
            all_tokens.append(tokens)
    
    all_tokens = torch.tensor(all_tokens, dtype=torch.long)
    all_masks = torch.tensor(all_masks, dtype=torch.float)
    dataset = TensorDataset(all_tokens, all_masks)

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)
    
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True), all_labels

def evaluate_ocnli(model, dev_dataloader, device, args):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dev_dataloader):
            tokens_1, masks_1, tokens_2, masks_2, tokens_3, masks_3, labels = [x.to(device) for x in batch]

            tokens, attention_mask, position_ids = get_batch(tokens_1, args)
            output, _ = model(tokens, position_ids, attention_mask)

            losses = mpu.vocab_parallel_cross_entropy(output[:, :-1, :].contiguous().float(), tokens[:, 1:])

            output_1 = torch.sum(losses * masks_1, 1) / torch.sum(masks_1, -1)

            tensor_list = [torch.zeros_like(output_1) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(tensor_list, output_1, mpu.get_data_parallel_group())
            output_1 = torch.stack(tensor_list, 0).view(-1).cpu().detach().numpy()

            # --------------
            tokens, attention_mask, position_ids = get_batch(tokens_2, args)
            output, _ = model(tokens, position_ids, attention_mask)
            losses = mpu.vocab_parallel_cross_entropy(output[:, :-1, :].contiguous().float(), tokens[:, 1:])

            output_2 = torch.sum(losses * masks_2, 1) / torch.sum(masks_2, -1)

            tensor_list = [torch.zeros_like(output_2) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(tensor_list, output_2, mpu.get_data_parallel_group())
            output_2 = torch.stack(tensor_list, 0).view(-1).cpu().detach().numpy()

            # ---------------

            tokens, attention_mask, position_ids = get_batch(tokens_3, args)
            output, _ = model(tokens, position_ids, attention_mask)
            losses = mpu.vocab_parallel_cross_entropy(output[:, :-1, :].contiguous().float(), tokens[:, 1:])

            output_3 = torch.sum(losses * masks_3, 1) / torch.sum(masks_3, -1)

            tensor_list = [torch.zeros_like(output_3) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(tensor_list, output_3, mpu.get_data_parallel_group())
            output_3 = torch.stack(tensor_list, 0).view(-1).cpu().detach().numpy()


            # --------------

            tensor_list_labels = [torch.zeros_like(labels) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(tensor_list_labels, labels, mpu.get_data_parallel_group())

            if torch.distributed.get_rank() == 0:
                labels = torch.stack(tensor_list_labels, 0)
                labels = labels.view(-1).cpu().detach().numpy()
                res = [np.argmin(np.array(x)) for x in zip(output_1, output_2, output_3)]
                res = [x==y for x, y in zip(res, labels)]
                correct += sum(res)
                total += len(res)
    
    if torch.distributed.get_rank() == 0:
        print("EVAL", correct, total)

def evaluate(model, dev_dataloader, all_labels, device, args):
    model.eval()

    if torch.distributed.get_rank() == 0:
        res = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dev_dataloader):
            tokens, masks = [x.to(device) for x in batch]

            tokens, attention_mask, position_ids = get_batch(tokens, args)
            output, _ = model(tokens, position_ids, attention_mask)
            losses = mpu.vocab_parallel_cross_entropy(output[:, :-1, :].contiguous().float(), tokens[:, 1:])

            output = torch.sum(losses * masks, 1) / torch.sum(masks, -1)

            tensor_list = [torch.zeros_like(output) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(tensor_list, output, mpu.get_data_parallel_group())
            output = torch.stack(tensor_list, 0).view(-1).cpu().detach().numpy()

            if torch.distributed.get_rank() == 0:
                for v in output:
                    res.append(v)

    if torch.distributed.get_rank() == 0:
        cnt = 0
        label_size = max(all_labels) + 1
        num_inst = len(res) // label_size
        for x in range(num_inst):
            label = all_labels[x]
            cur_res = res[x*label_size:(x+1)*label_size]
            pos = np.argmin(cur_res)
            if pos == label:
                cnt += 1
        print("EVAL", cnt, num_inst)

def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    #get the tokenizer
    tokenizer = GPT2Tokenizer(os.path.join(args.tokenizer_path, 'vocab.json'), os.path.join(args.tokenizer_path, 'chinese_vocab.model'))

    # load data
    assert args.eval_data_path is not None

    device = torch.cuda.current_device()
    args.eod_token = tokenizer.encoder['<eod>']

    # Model
    args.parallel_output = True
    model = setup_model(args)

    if args.task == "ocnli":
        dev_dataloader = load_ocnli_data(args.eval_data_path, 'dev', tokenizer)
        evaluate_ocnli(model, dev_dataloader, device, args)
    elif args.task == "iflytek":
        dev_dataloader, all_labels = load_iflytek_data(args.eval_data_path, 'dev', tokenizer, True)
        evaluate(model, dev_dataloader, all_labels, device, args)
    elif args.task == "tnews":
        dev_dataloader, all_labels = load_tnews_data(args.eval_data_path, 'dev', tokenizer, True)
        evaluate(model, dev_dataloader, all_labels, device, args)
    else:
        print("Unknown task!")

if __name__ == "__main__":
    main()
