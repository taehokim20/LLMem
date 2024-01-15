import torch.nn.functional as F
from typing import Optional
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ComputePattern, ColoTensorSpec, ComputePattern, ComputeSpec, ColoTensor, ShardSpec, \
    ReplicaSpec
from ._utils import GeneralTensor, convert_to_colo_tensor, reduce_input
import torch.distributed as dist
import torch
from colossalai.utils import print_rank_0
import sys


def colo_embedding_2p5d(input_tensor:ColoTensor,
                      weight: ColoTensor,
                      padding_idx: Optional[int] = None,
                      max_norm: Optional[float] = None,
                      norm_type: float = 2.0,
                      scale_grad_by_freq: bool = False,
                      sparse: bool = False) -> ColoTensor:
    pass


def colo_embedding_2d(input_tensor:ColoTensor,
                      weight: ColoTensor,
                      padding_idx: Optional[int] = None,
                      max_norm: Optional[float] = None,
                      norm_type: float = 2.0,
                      scale_grad_by_freq: bool = False,
                      sparse: bool = False) -> ColoTensor:
    local_rank = dist.get_rank()
    pg = weight.get_process_group()
    output_replicate = weight.compute_spec.output_replicate
    summa = 2 #int(math.sqrt(weight.get_tp_world_size()))
    if int(local_rank / summa) == 0:
        input_tensor = input_tensor[0:int(input_tensor.size()[0]/summa)]
    else:
        input_tensor = input_tensor[int(input_tensor.size()[0]/summa):int(input_tensor.size()[0])]
    # If the input is partitioned by 4
    # isize = [input_tensor.size()[0], input_tensor.size()[1]*2]
    # partition_input = torch.zeros(isize, dtype=input_tensor.dtype, device=input_tensor.device)
    # dist.all_gather_into_tensor(partition_input, input_tensor, group=dist.new_group(ranks=[0, 1]))
    # dist.all_gather_into_tensor(partition_input, input_tensor, group=dist.new_group(ranks=[2, 3]))
    # input_tensor = partition_input

    # ### Trial 1. Only communicating the required part (partial area) 
    # ### 1-1. dist.all_gather_into_tensor
    # pg = weight.get_process_group()
    # output_replicate = weight.compute_spec.output_replicate
    # wsize = [weight.data.size()[0]*2, weight.data.size()[1]]
    # weight_out = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
    # dist.all_gather_into_tensor(weight_out, weight.data, group=dist.new_group(ranks=[0, 2])) 
    # dist.all_gather_into_tensor(weight_out, weight.data, group=dist.new_group(ranks=[1, 3])) 

    ### 1-2. dist.broadcast or point-to-point
    wsize = [weight.data.size()[0], weight.data.size()[1]]
    if local_rank == 0:
        weight0 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        weight0.copy_(weight)
        weight2 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        # dist.broadcast(weight0, local_rank, group=dist.new_group(ranks=[0, 2]))
        dist.send(tensor=weight0, dst=2)
        dist.recv(tensor=weight2, src=2)
        weight = torch.vstack((weight0, weight2))
    elif local_rank == 1:
        weight1 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        weight1.copy_(weight)
        weight3 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        # dist.broadcast(weight1, local_rank, group=dist.new_group(ranks=[1, 3]))
        dist.send(tensor=weight1, dst=3)
        dist.recv(tensor=weight3, src=3)
        weight = torch.vstack((weight1, weight3))
    elif local_rank == 2:
        weight0 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        weight2 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        weight2.copy_(weight)
        # dist.broadcast(weight2, local_rank, group=dist.new_group(ranks=[0, 2]))
        dist.recv(tensor=weight0, src=0)
        dist.send(tensor=weight2, dst=0)
        weight = torch.vstack((weight0, weight2))
    else:
        weight1 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        weight3 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        weight3.copy_(weight)
        # dist.broadcast(weight3, local_rank, group=dist.new_group(ranks=[1, 3]))
        dist.recv(tensor=weight1, src=1)
        dist.send(tensor=weight3, dst=1)
        weight = torch.vstack((weight1, weight3))

    # ### Trial 2. Inefficient all_gather communication
    # pg = weight.get_process_group()
    # output_replicate = weight.compute_spec.output_replicate
    # all_weights = weight.to_replicate()
    # if int(local_rank % summa) == 0:
    #     weight = all_weights[:, 0:int(all_weights.size()[1]/summa)]
    # else:
    #     weight = all_weights[:, int(all_weights.size()[1]/summa):all_weights.size()[1]]

    output_parallel = F.embedding(input_tensor,
                                  weight, #weight_out,
                                  padding_idx=padding_idx,
                                  max_norm=max_norm,
                                  norm_type=norm_type,
                                  scale_grad_by_freq=scale_grad_by_freq,
                                  sparse=sparse)
    
    output_spec = ColoTensorSpec(pg, ShardSpec([0, -1], [summa, summa]), ComputeSpec(ComputePattern.TP2D))
    output = ColoTensor.from_torch_tensor(output_parallel, spec=output_spec)

    if output_replicate:
        return output.to_replicate()
    else:
        return output


def colo_embedding_1Dcol(input_tensor: ColoTensor,
                         weight: ColoTensor,
                         padding_idx: Optional[int] = None,
                         max_norm: Optional[float] = None,
                         norm_type: float = 2.0,
                         scale_grad_by_freq: bool = False,
                         sparse: bool = False) -> ColoTensor:
    # embedding_1Dcol split the weight(lookup table) to (num_embeddings, embedding_dim/P)
    # Gather splitted lookup table
    # input_tensor = input_tensor.redistribute(ReplicaSpec())

    # 1st output_parallel size: [8, 169, 512]
    output_parallel = F.embedding(input_tensor,    # 1st input_tensor size: [8, 169]
                                  weight,          # weight.data.size(): [32000, 512]
                                  padding_idx=padding_idx,
                                  max_norm=max_norm,
                                  norm_type=norm_type,
                                  scale_grad_by_freq=scale_grad_by_freq,
                                  sparse=sparse)
    # output_spec
    # ColoTensorSpec(pg=ProcessGroup(ranks=[0, 1, 2, 3, 4, 5, 6, 7], rank=0, dp=1, tp=8),
    #                dist_attr=DistSpec(dims=(-1,), num_partitions=(8,), placement=DistPlacementPattern.SHARD), 
    #                compute_attr=ComputeSpec(pattern=ComputePattern.TP1D, replicate_output=True))
    output_spec = ColoTensorSpec(weight.get_process_group(), ShardSpec([-1], [weight.get_tp_world_size()]),
                                 ComputeSpec(ComputePattern.TP1D))
    output = ColoTensor.from_torch_tensor(output_parallel, spec=output_spec)
    output.to_replicate()

    # weight.compute_spec
    # ComputeSpec(pattern=ComputePattern.TP1D, replicate_output=True)
    compute_spec = weight.compute_spec

    if compute_spec.output_replicate:
        return output.to_replicate()   # [8, 169, 4096]
    else:
        return output


def colo_embedding_1Drow(input_tensor: ColoTensor,
                         weight: ColoTensor,
                         padding_idx: Optional[int] = None,
                         max_norm: Optional[float] = None,
                         norm_type: float = 2.0,
                         scale_grad_by_freq: bool = False,
                         sparse: bool = False) -> ColoTensor:
    # embedding_1Drow splits the weight(lookup table) to the shape, [num_embeddings/P, embedding_dim]
    # get the index of current segment and mask other segments with 0

    # get complete input tensor through all-gather
    input_tensor = input_tensor.redistribute(ReplicaSpec())

    # tensor_parallel_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
    tensor_parallel_rank = weight.get_process_group().tp_local_rank()
    num_embeddings_per_partition = weight.size_local(0)
    vocab_start_index = tensor_parallel_rank * num_embeddings_per_partition
    vocab_end_index = vocab_start_index + num_embeddings_per_partition

    # build the mask.
    input_mask = (input_tensor < vocab_start_index) | (input_tensor >= vocab_end_index)
    # mask the input.
    # TODO(jzy) masked_input may be an activation managed by ColoTensor.
    masked_input = input_tensor - vocab_start_index
    masked_input[input_mask] = 0

    partial_output = F.embedding(masked_input,
                                 weight,
                                 padding_idx=padding_idx,
                                 max_norm=max_norm,
                                 norm_type=norm_type,
                                 scale_grad_by_freq=scale_grad_by_freq,
                                 sparse=sparse)

    # Mask the output embedding.
    partial_output[input_mask, :] = 0.
    # Reduce across all the model parallel GPUs.
    output = reduce_input(partial_output, weight.get_process_group())
    output = ColoTensor.from_torch_tensor(output, spec=ColoTensorSpec(weight.get_process_group(), ReplicaSpec()))

    return output


def colo_embedding_1d(mode: str,
                      input_tensor: ColoTensor,
                      weight: ColoTensor,
                      padding_idx: Optional[int] = None,
                      max_norm: Optional[float] = None,
                      norm_type: float = 2.0,
                      scale_grad_by_freq: bool = False,
                      sparse: bool = False) -> ColoTensor:
    assert mode in ('row', 'col')
    funcs = {'row': colo_embedding_1Drow, 'col': colo_embedding_1Dcol}
    return funcs[mode](input_tensor,
                       weight,
                       padding_idx=padding_idx,
                       max_norm=max_norm,
                       norm_type=norm_type,
                       scale_grad_by_freq=scale_grad_by_freq,
                       sparse=sparse)


@colo_op_impl(F.embedding)
def colo_embedding(input_tensor: GeneralTensor,
                   weight: GeneralTensor,
                   padding_idx: Optional[int] = None,
                   max_norm: Optional[float] = None,
                   norm_type: float = 2.0,
                   scale_grad_by_freq: bool = False,
                   sparse: bool = False):
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding``.
    This method looks up an embedding table.
    """
    assert isinstance(weight, ColoTensor)
    input_tensor = convert_to_colo_tensor(input_tensor, weight.get_process_group())

    if not weight.has_compute_spec():    # No Model Parallel Applied
        assert weight.is_replicate(), 'Invalid weight spec for native embedding op'
        return ColoTensor.from_torch_tensor(tensor=F.embedding(input_tensor,
                                                               weight,
                                                               padding_idx=padding_idx,
                                                               max_norm=max_norm,
                                                               norm_type=norm_type,
                                                               scale_grad_by_freq=scale_grad_by_freq,
                                                               sparse=sparse),
                                            spec=ColoTensorSpec(weight.get_process_group()))
    elif weight.has_compute_pattern(ComputePattern.TP1D):    # Single Model Parallel Applied
        if weight.is_shard_1drow():
            mode = 'row'
        elif weight.is_shard_1dcol():
            mode = 'col'
        else:
            raise NotImplementedError
        return colo_embedding_1d(mode,
                                 input_tensor,
                                 weight,
                                 padding_idx=padding_idx,
                                 max_norm=max_norm,
                                 norm_type=norm_type,
                                 scale_grad_by_freq=scale_grad_by_freq,
                                 sparse=sparse)
    elif weight.has_compute_pattern(ComputePattern.TP2D):
        return colo_embedding_2d(input_tensor,
                                 weight,
                                 padding_idx=padding_idx,
                                 max_norm=max_norm,
                                 norm_type=norm_type,
                                 scale_grad_by_freq=scale_grad_by_freq,
                                 sparse=sparse)
    elif weight.has_compute_pattern(ComputePattern.TP2P5D):
        return colo_embedding_2p5d(input_tensor,
                                 weight,
                                 padding_idx=padding_idx,
                                 max_norm=max_norm,
                                 norm_type=norm_type,
                                 scale_grad_by_freq=scale_grad_by_freq,
                                 sparse=sparse)
    else:
        raise NotImplementedError
