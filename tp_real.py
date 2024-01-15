#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import warnings
warnings.simplefilter("ignore", UserWarning)

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import get_cosine_schedule_with_warmup

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.tensor import ProcessGroup, ShardSpec
from colossalai.utils import get_current_device, print_rank_0
from colossalai.zero import ColoInitContext
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.cluster import DistCoordinator
from colossalai.context import ParallelMode
import torch.distributed as dist
from tqdm import tqdm
from colossalai.tensor import ProcessGroup, ShardSpec, ComputePattern, ComputeSpec
from colossalai.core import global_context as gpc
from statistics import mean
import GPUtil
import psutil
from pynvml import *
from size_estimator import SizeEstimator

from transformers import AutoConfig

import transformers.models.llama.modeling_llama # LLaMA
import transformers.models.opt.modeling_opt # OPT
import transformers.models.gpt2.modeling_gpt2 # GPT2
import transformers.models.bloom.modeling_bloom # BLOOM
import transformers.models.biogpt.modeling_biogpt # biogpt
import transformers.models.codegen.modeling_codegen # codegen
import transformers.models.gpt_bigcode.modeling_gpt_bigcode # gpt_bigcode-santacoder
import transformers.models.gpt_neo.modeling_gpt_neo # gpt-neo-1.3B

txt_file_name = 'temp.txt'
max_seq_len = 512
tp_size = 2

def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator, tp_dim=0, 
                batch_size=1, tp_degree=None, dims=None):
    torch.cuda.synchronize()
    model.train()
    losses = []
    peak_gpu = 0
    cnt = 0
    local_rank = dist.get_rank()
    with tqdm(dataloader, desc=f'Epoch [{epoch + 1}]', disable=not coordinator.is_master()) as pbar:
        for batch in pbar:
            # Forward
            batch = move_to_cuda(batch, torch.cuda.current_device())
            if batch["input_ids"].size()[1] < max_seq_len: #!= 64: #< 450:
                continue
            model.zero_grad()
            a = GPUtil.getGPUs()[local_rank].memoryUsed
            outputs = model(use_cache=False, **batch)  ## cf. outputs['logits']
            b = GPUtil.getGPUs()[local_rank].memoryUsed
            loss = outputs['loss']
            outputs = 0
            # Backward
            booster.backward(loss, optimizer)
            c = GPUtil.getGPUs()[local_rank].memoryUsed
            if cnt == 0:
                torch.cuda.empty_cache()
            # optimizer.clip_grad_by_norm(max_norm=0.1) # GRADIENT_CLIPPING=0.1
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            d = GPUtil.getGPUs()[local_rank].memoryUsed
            # Print batch loss
            cur_gpu_mem = d
            if cur_gpu_mem > peak_gpu:
                peak_gpu = cur_gpu_mem
            pbar.set_postfix({'loss': loss.item(), 'peak': peak_gpu})
            # pbar.set_postfix({'loss': loss.item()}) 
            losses.append(loss.item())
            if local_rank == 0:
                with open(txt_file_name, 'a') as f:
                    # f.write('[next]: {} -> {} --- {} -> {} --- {} -> {}\n\n'.format(a, b, b2, c, c2, d))
                    f.write('[cuda:{} - {}]: {} -> {} -> {} -> {}\n'.format(local_rank, cnt, a, b, c, d))
            elif local_rank == 1:
                with open(txt_file_name, 'a') as f:
                    f.write('[cuda:{} - {}]: {} -> {} -> {} -> {}\n'.format(local_rank, cnt, a, b, c, d))
            elif local_rank == 2:
                with open(txt_file_name, 'a') as f:
                    f.write('[cuda:{} - {}]: {} -> {} -> {} -> {}\n'.format(local_rank, cnt, a, b, c, d))
            elif local_rank == 3:
                with open(txt_file_name, 'a') as f:
                    f.write('[cuda:{} - {}]: {} -> {} -> {} -> {}\n'.format(local_rank, cnt, a, b, c, d))
            cnt += 1
            if cnt > 8:
                break

    # if dist.get_rank() == 0:
    #     import time
    #     time.sleep(3)
    #     with open(txt_file_name, 'a') as f:
    #         f.write('\npeak: {} MB\n'.format(peak_gpu))
                    
    # print_rank_0('Average loss of epoch {0}: {1:.2f}, Memory usage: {2}'.format(epoch + 1, mean(losses), 
    #                                                                             GPUtil.getGPUs()[0].memoryUsed))


IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=max_seq_len, #512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format- KB, MB, GB, TB and PB
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def train():
    norm_sharding=False
    tp_dim=0 # 0: 1D TP, 1: 2D TP, 2: 2.5D TP, 3: 3D TP
    mode=['1d', '2d', '2.5d', '3d']
    tp_degree=[[tp_size], [2, 2], [2, 2, 2], [2, 2, 2]]
    dims_e=[[-1], [0, -1], [0, 0, -1], [0, 0, -1]]
    dims_l=[[-1], [0, -1], [0, 0, -1], [0, 0, -1]]
    # Compute Pattern
    compute_spec = [ComputeSpec(ComputePattern.TP1D), ComputeSpec(ComputePattern.TP2D),
                    ComputeSpec(ComputePattern.TP2P5D), ComputeSpec(ComputePattern.TP3D)]
    # Launch ColossalAI - Tensor Parallelism
    if mode == '2.5d':
        colossalai.launch_from_torch(config=dict(parallel=dict(data=1, pipeline=1,
                                    tensor=dict(size=tp_size, mode=mode[tp_dim], depth=2))))
    else:
        colossalai.launch_from_torch(config=dict(parallel=dict(data=1, pipeline=1, 
                                    tensor=dict(size=tp_size, mode=mode[tp_dim]))))
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    coordinator = DistCoordinator()
    world_size = coordinator.world_size
    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()

    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    total_nvml = int(info.total / (1024 * 1024))
    used_nvml = int(info.used / (1024 * 1024))
    print_rank_0('[0]Total nvml GPU mem: {}'.format(total_nvml))
    print_rank_0('[0]Used nvml GPU mem: {}'.format(used_nvml))
    cuda_context_mem = used_nvml - GPUtil.getGPUs()[dist.get_rank()].memoryUsed
    framework_initial_mem = GPUtil.getGPUs()[dist.get_rank()].memoryUsed
    print_rank_0('[0]Used GPUtil GPU mem: {}'.format(framework_initial_mem))

    shard_pg = ProcessGroup(tp_degree=tp_size)

    with ColoInitContext(device=get_current_device()):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        ).to('cuda')        
        model.half() # cf. PRECISION_STR_TO_DTYPE = {'fp16': torch.half, 'bf16': torch.bfloat16}
        torch.cuda.empty_cache()

        model.gradient_checkpointing_enable()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.eos_token is None:                 # for bert
            tokenizer.eos_token = tokenizer.pad_token     #

        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)           

    compute_spec = ComputeSpec(ComputePattern.TP1D)
    from colossalai.nn.parallel.layers import init_colo_module
    init_colo_module(model, compute_spec, pg=shard_pg, recursive=True)
    torch.cuda.empty_cache()

    # Set plugin
    booster_kwargs = {}
    plugin = GeminiPlugin(device=get_current_device(),
                          placement_policy='cuda',
                          precision='fp16',
                          pin_memory=False,
                          strict_ddp_mode=False,
                          initial_scale=2**5)

    config = {
        'batch_size': training_args.per_device_train_batch_size,
        'lr': training_args.learning_rate,
        'epochs': int(training_args.num_train_epochs),
        'warmup_ratio': training_args.warmup_ratio,
        'weight_decay': training_args.weight_decay,
    }

    dataloader = plugin.prepare_dataloader(data_module['train_dataset'], batch_size=config['batch_size'],
                                           shuffle=False, drop_last=True, collate_fn=data_module['data_collator'])
    
    for batch in dataloader:
        if batch["input_ids"].size()[1] == max_seq_len:
            test_long_input = move_to_cuda(batch, torch.cuda.current_device())
            break

    # Set lr scheduler
    total_steps = len(dataloader) * config['epochs']
    num_warmup_steps = int(config['warmup_ratio'] * total_steps)
        
    # Set optimizer
    optimizer = HybridAdam(model.parameters(), lr=config['lr'], weight_decay=0.0)

    # Set lr scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(dataloader) * config['epochs']
    )
    
    # ############################## 1 ##############################
    # lm_fp32 = False
    # if ('codegen' in model_args.model_name_or_path) or ('neo' in model_args.model_name_or_path):
    #     lm_fp32 = True
    # real_bs = 0 # For batch size search mode 
    # # real_bs = test_long_input["input_ids"].size()[0] # To estimate with specific batch size
    # se = SizeEstimator(model, test_long_input["input_ids"][0:2], real_bs, bytes=2, bytes_input=8,  
    #                    gpu_n=world_size, tp=tp_size, lm_fp32=lm_fp32, m_total=total_nvml)
    # torch.cuda.empty_cache()
    # prev_get_output = GPUtil.getGPUs()[dist.get_rank()].memoryUsed
    # se.get_output_sizes()
    # torch.cuda.empty_cache()
    # after_get_output = GPUtil.getGPUs()[dist.get_rank()].memoryUsed
    # ###############################################################

    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, _, _ = booster.boost(model, optimizer)
    torch.cuda.empty_cache()

    # ############################## 2 ##############################
    # booster_chunk_mem = GPUtil.getGPUs()[dist.get_rank()].memoryUsed
    # m_pbase = booster_chunk_mem + cuda_context_mem - (after_get_output - prev_get_output)
    # print_rank_0('[m_pbase]: {}'.format(m_pbase))
    # esti_mem, real_bs = se.estimate_size(m_init=m_pbase)
    # print_rank_0('Estimated memory: {0}, real bs: {1}'.format(esti_mem, real_bs))
    # torch.cuda.empty_cache()
    # import sys
    # sys.exit()
    # ###############################################################

    # with open(txt_file_name, 'a') as f:
    #     f.write('[cuda:{}] Before fine-tuning: {} -> {} -> {}\n'.format(dist.get_rank(),
    #                         framework_initial_mem, load_model_mem, booster_chunk_mem))

    # Start finetuning
    logger.info(f"Start finetuning", ranks=[0])
    for epoch in range(config['epochs']):
       train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator, 
                   tp_dim=tp_dim, batch_size=config['batch_size']) #tp_degree=tp_degree, dims=dims_l) 

    # Finish training and evaluate
    logger.info(f"Finish finetuning", ranks=[0])
    # output_dir = training_args.output_dir + '/shard_' + str(dist.get_rank()) + '.pt'
    # booster.save_model(model, output_dir, tp_degree=world_size)
    # logger.info(f"Saving model checkpoint to {output_dir}")


if __name__ == "__main__":
    train()
