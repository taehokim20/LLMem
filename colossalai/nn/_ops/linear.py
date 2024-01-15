from copy import deepcopy
from typing import Optional

import torch.nn.functional as F

from colossalai.tensor import ColoTensor, ColoTensorSpec, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor.sharding_spec import ShardingSpec

from ._utils import GeneralTensor, convert_to_colo_tensor, reduce_grad, reduce_input
import torch.distributed as dist
import torch
from colossalai.utils import print_rank_0
import sys
import time


def colo_linear_2d(input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> 'ColoTensor':
    # input_tensor = input_tensor.redistribute(ReplicaSpec())
    local_rank = dist.get_rank()
    pg = weight.get_process_group()
    output_replicate = weight.compute_spec.output_replicate
    summa = 2 #int(math.sqrt(weight.get_tp_world_size()))
    if local_rank == 0:
        partial_input = input_tensor[0:int(input_tensor.size()[0]/summa), :, 0:int(input_tensor.size()[2]/summa)]
    elif local_rank == 1:
        partial_input = input_tensor[0:int(input_tensor.size()[0]/summa), :, 
                                    int(input_tensor.size()[2]/summa):int(input_tensor.size()[2])]
    elif local_rank == 2:
        partial_input = input_tensor[int(input_tensor.size()[0]/summa):int(input_tensor.size()[0]), :, 
                                    0:int(input_tensor.size()[2]/summa)]
    else:
        partial_input = input_tensor[int(input_tensor.size()[0]/summa):int(input_tensor.size()[0]), :, 
                                    int(input_tensor.size()[2]/summa):int(input_tensor.size()[2])]

    ### Trial 0. Point-to-point
    wsize = [weight.data.size()[0], weight.data.size()[1]]
    if dist.get_rank() == 0:
        weight0 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        weight0.copy_(weight)
        weight2 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        dist.send(tensor=weight0, dst=2)
        dist.recv(tensor=weight2, src=2)
        weight = torch.vstack((weight0, weight2))
    elif local_rank == 1:
        weight1 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        weight1.copy_(weight)
        weight3 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        dist.send(tensor=weight1, dst=3)
        dist.recv(tensor=weight3, src=3)
        weight = torch.vstack((weight1, weight3))
    elif local_rank == 2:
        weight0 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        weight2 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        weight2.copy_(weight)
        dist.recv(tensor=weight0, src=0)
        dist.send(tensor=weight2, dst=0)
        weight = torch.vstack((weight0, weight2))
    else:
        weight1 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        weight3 = torch.zeros(wsize, dtype=weight.data.dtype, device=weight.data.device)
        weight3.copy_(weight)
        dist.recv(tensor=weight1, src=1)
        dist.send(tensor=weight3, dst=1)
        weight = torch.vstack((weight1, weight3))

    # # ### Trial 1. all_gather partially
    # pg = weight.get_process_group()
    # weight_list = [torch.empty_like(weight.data) for _ in range(summa)]
    # weight_list[int(local_rank % summa)] = weight.data
    # dist.all_gather(weight_list, weight.data, group=dist.new_group(ranks=[0, 1]))
    # dist.all_gather(weight_list, weight.data, group=dist.new_group(ranks=[2, 3]))
    # weight = torch.cat(weight_list, -1)
    # partial_output = F.linear(input_tensor, weight)
    # output_spec = ColoTensorSpec(pg, ShardSpec([-1, 0], [summa, summa]), ComputeSpec(ComputePattern.TP2D))
    # output = ColoTensor.from_torch_tensor(partial_output, spec=output_spec)
    # # output = output.to_replicate()
    # # output.set_dist_spec(ShardSpec([0, -1], [summa, summa]))

    # # ### Trial 2. broadcast
    # pg = weight.get_process_group()
    # temp = weight.data
    # # 0, 1 -> [0, 2], 2, 3 -> [1, 3]
    # if int(local_rank / summa) == 0:
    #     temp_group = dist.new_group(ranks=[0, 2])
    # else:
    #     temp_group = dist.new_group(ranks=[1, 3])
    # dist.broadcast(temp, src=local_rank, group=temp_group)
    # weight = torch.cat([weight.data, temp], -1)
    # partial_output = F.linear(input_tensor, weight)
    # output_spec = ColoTensorSpec(pg, ShardSpec([0, -1], [summa, summa]), ComputeSpec(ComputePattern.TP2D))
    # output = ColoTensor.from_torch_tensor(partial_output, spec=output_spec)

    # ### Trial 3. Inefficient all_gather communication
    # pg = weight.get_process_group()
    # all_weight = weight.to_replicate()
    # if int(local_rank % summa) == 0:
    #     partial_weight = all_weight[0:int(all_weight.size()[0]/summa), :]
    # else:
    #     partial_weight = all_weight[int(all_weight.size()[0]/summa):all_weight.size()[0], :]
    
    partial_output = F.linear(partial_input, weight)
    wsize = [partial_output.data.size()[0], partial_output.data.size()[1], int(partial_output.data.size()[2]/summa)]
    if local_rank == 0:
        temp0 = torch.zeros(wsize, dtype=partial_output.data.dtype, device=partial_output.data.device)
        temp1 = torch.zeros(wsize, dtype=partial_output.data.dtype, device=partial_output.data.device)
        temp1.copy_(partial_output[:, :, int(partial_output.size()[2]/summa):int(partial_output.size()[2])])
        dist.send(tensor=temp1, dst=1)
        dist.recv(tensor=temp0, src=1)
        partial_output = temp0 + partial_output[:, :, 0:int(partial_output.size()[2]/summa)]
    elif local_rank == 1:
        temp0 = torch.zeros(wsize, dtype=partial_output.data.dtype, device=partial_output.data.device)
        temp0.copy_(partial_output[:, :, 0:int(partial_output.size()[2]/summa)])
        temp1 = torch.zeros(wsize, dtype=partial_output.data.dtype, device=partial_output.data.device)
        dist.recv(tensor=temp1, src=0)
        dist.send(tensor=temp0, dst=0)
        partial_output = temp1 + partial_output[:, :, int(partial_output.size()[2]/summa):int(partial_output.size()[2])]
    elif local_rank == 2:
        temp2 = torch.zeros(wsize, dtype=partial_output.data.dtype, device=partial_output.data.device)
        temp3 = torch.zeros(wsize, dtype=partial_output.data.dtype, device=partial_output.data.device)
        temp3.copy_(partial_output[:, :, int(partial_output.size()[2]/summa):int(partial_output.size()[2])])
        dist.send(tensor=temp3, dst=3)
        dist.recv(tensor=temp2, src=3)
        partial_output = temp2 + partial_output[:, :, 0:int(partial_output.size()[2]/summa)]
    else:
        temp2 = torch.zeros(wsize, dtype=partial_output.data.dtype, device=partial_output.data.device)
        temp2.copy_(partial_output[:, :, 0:int(partial_output.size()[2]/summa)])
        temp3 = torch.zeros(wsize, dtype=partial_output.data.dtype, device=partial_output.data.device)
        dist.recv(tensor=temp3, src=2)
        dist.send(tensor=temp2, dst=2)
        partial_output = temp3 + partial_output[:, :, int(partial_output.size()[2]/summa):int(partial_output.size()[2])]

    output_spec = ColoTensorSpec(pg, ShardSpec([0, -1], [summa, summa]), ComputeSpec(ComputePattern.TP2D))
    output = ColoTensor.from_torch_tensor(partial_output, spec=output_spec)
    
    #return output.to_replicate()
    if output_replicate:
        return output.to_replicate()  # [8, 169, 4096]
    else:
        return output


def colo_linear_1drow(input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> 'ColoTensor':
    # Input:S[1] x Weight:S[0] = Output:P
    # All-Reduce(Output) + bias = res
    # Input:S[1]
    pg = weight.get_process_group()
    input_tensor = input_tensor.redistribute(ShardSpec([-1], [weight.get_tp_world_size()]), pg) # [8, 169, 4096] -> [8, 169, 512]

    # Output:P
    partial_output = F.linear(input_tensor, weight) # input_tensor: [8, 169, 512], weight: [4096, 512],
                                                    # partial_output: [8, 169, 4096]
    # Reduce(Output)

    output = reduce_input(partial_output, pg) # output: [8, 169, 4096]

    # Bias
    if bias is not None:
        assert not bias.has_compute_spec(), 'Invalid bias spec for 1Drow Linear op'
        output = output + bias

    output = ColoTensor.from_torch_tensor(output, spec=ColoTensorSpec(pg, ReplicaSpec()))

    return output


def colo_linear_1dcol(input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> 'ColoTensor':
    # Input:B x Weight:S[1] + Bias:S[1] = Output:S[1]
    # All-Gather(Output)
    # Input:B
    import GPUtil
    compute_spec = weight.compute_spec
    input_tensor = input_tensor.redistribute(ReplicaSpec())  # [8, 169, 4096], [8, 169, 4096]
    input_parallel = reduce_grad(input_tensor, weight.get_process_group())  # [8, 169, 4096], [8, 169, 4096]
    # input_parallel = input_tensor

    output_parallel = F.linear(input_parallel, weight, bias)  # input_parallel: [8, 169, 4096], weight: [512, 4096]
                                                              # output_parallel: [8, 169, 512]
    output = ColoTensor.from_torch_tensor(output_parallel,
                                          spec=ColoTensorSpec(weight.get_process_group(),
                                                              ShardSpec([-1], [weight.get_tp_world_size()]),
                                                              ComputeSpec(ComputePattern.TP1D)))  # [8, 169, 512]

    if compute_spec.output_replicate:
        return output.to_replicate()  # [8, 169, 4096]
    else:
        return output


def colo_linear_1d(mode: str, input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> 'ColoTensor':
    assert mode in ('row', 'col')
    funcs = {'row': colo_linear_1drow, 'col': colo_linear_1dcol}
    return funcs[mode](input_tensor, weight, bias)


# @register_colo_graph(input_pos=[1], param_pos=[2, 3])
def colo_linear_imp(input_tensor: GeneralTensor,  # 1st input_tensor size: [8, 169, 4096]
                    weight: GeneralTensor,        # 1st weight size: [4096, 512]
                    bias: Optional[GeneralTensor] = None) -> 'ColoTensor':
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.linear``.
    This method computes a linear.
    """
    assert isinstance(weight, ColoTensor)
    pg = weight.get_process_group()
    assert pg
    input_tensor = convert_to_colo_tensor(input_tensor, pg)
    bias = convert_to_colo_tensor(bias, pg)
    # input_tensor, weight, bias = tuple(map(convert_to_colo_tensor, (input_tensor, weight, bias)))

    # Add communication logic before and after linear call.
    ret_tensor = None
    if not weight.has_compute_spec():    # No Model Parallel Applied
        assert weight.is_replicate(), 'Invalid weight spec for native Linear op'
        assert bias is None or bias.is_replicate(), 'Invalid bias spec for native Linear op'
        ret_tensor = ColoTensor.from_torch_tensor(F.linear(input_tensor, weight, bias), spec=ColoTensorSpec(pg))
    elif weight.has_compute_pattern(ComputePattern.TP1D):    # Single Model Parallel Applied
        if weight.is_shard_1dcol() and (bias is None or bias.is_replicate()):
            mode = 'row'
        elif weight.is_shard_1drow() and (bias is None or bias.is_shard_1drow() or bias.is_shard_1dcol()):
            mode = 'col'
        else:
            raise RuntimeError(f"the weight or bias tensor spec is not valid, weight {weight}, bias {bias}")
        ret_tensor = colo_linear_1d(mode, input_tensor, weight, bias)
    elif weight.has_compute_pattern(ComputePattern.TP2D):
        ret_tensor = colo_linear_2d(input_tensor, weight, bias)
    else:
        raise NotImplementedError

    return ret_tensor


def _new_colo_linear_imp(input_tensor: GeneralTensor,
                         weight: GeneralTensor,
                         bias: Optional[GeneralTensor] = None) -> 'ColoTensor':
    """
    A tentative function to compute the distributed linear layer with the latest sharding spec.
    This function is subject to future change as the current sharding API is not stable.
    """
    # get mesh info
    input_sharding_seq = input_tensor.sharding_spec.sharding_sequence
    weight_sharding_seq = weight.sharding_spec.sharding_sequence
    if bias is not None:
        bias_sharding_seq = bias.sharding_spec.sharding_sequence
    device_mesh = weight.sharding_spec.device_mesh
    pg_axis0 = weight.pg_axis0
    pg_axis1 = weight.pg_axis1

    # the last dim of input should have the same spec as the first dim of weight
    # the weight is transposed, so we look at the second dimension
    assert input_sharding_seq[-1] == weight_sharding_seq[1]

    if bias is not None:
        assert bias_sharding_seq[0] == weight_sharding_seq[0]

    # compute the output sharding sequence
    # as weight is transposed, so we look at the first dimension
    output_shard_seq = input_sharding_seq[:-1] + weight_sharding_seq[:1]
    output_shard_seq = deepcopy(output_shard_seq)

    # TODO: add reduce grad logic

    # handle column and row parallel linear
    # by reusing the implementation above
    out = F.linear(input_tensor, weight)

    # run all reduce if necessary
    last_dim_spec = input_sharding_seq[-1]
    if last_dim_spec.is_replica:
        pass
    elif last_dim_spec.shard_list is not None:
        for dim in last_dim_spec.shard_list:
            if dim == 0:
                reduce_input(out, pg_axis0)
            elif dim == 1:
                reduce_input(out, pg_axis1)
            else:
                raise RuntimeError("Found invalid sharding axis {dim}, only 0 or 1 is expected")
    # add bias
    if bias is not None:
        out += bias

    # convert shard seq to partition dict
    output_partition_dict = {}
    for index, dim_spec in enumerate(output_shard_seq):
        if not dim_spec.is_replica:
            if index not in output_partition_dict:
                output_partition_dict[index] = []
            output_partition_dict[index].extend(dim_spec.shard_list)

    entire_shape = out.shape
    output_sharding_spec = ShardingSpec(device_mesh, entire_shape, output_partition_dict)
    ret_tensor = ColoTensor.from_torch_tensor(out)
    setattr(ret_tensor, 'sharding_spec', output_sharding_spec)
    return ret_tensor


def _has_sharding_spec(tensor):
    """
    A tentative function to check whether the tensor is using the new sharding spec API. We assume that the sharding spec object is
    set as the attribute `sharding_spec` on a tensor.
    """
    return hasattr(tensor, 'sharding_spec')


@colo_op_impl(F.linear)
def colo_linear(input: GeneralTensor, weight: GeneralTensor, bias: Optional[GeneralTensor] = None) -> 'ColoTensor':
    if _has_sharding_spec(weight):
        return _new_colo_linear_imp(input, weight, bias)
    else:
        return colo_linear_imp(input, weight, bias)
