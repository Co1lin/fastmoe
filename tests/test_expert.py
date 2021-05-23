import sys
from collections import OrderedDict
from typing import List, Type, Union

import pytest
import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
from fmoe.gates import NaiveGate
from fmoe.layers import FMoE
from fmoe.transformer import _Expert
from fmoe.distributed import DistributedGroupedDataParallel as LocalDDP
from fmoe.megatron.layers import _megatron_init_method
from moe import BruteForceMoELinear, BruteForceMoE, NaiveExpert, LinearExpert

def _perform_forward(
        moe: nn.Module, moe_raw: nn.Module, batch_size, d_model, top_k, rank, mp_group, data_type='torch.FloatTensor'
):
    moe.zero_grad()
    moe_raw.zero_grad()

    inp = torch.rand(batch_size, d_model).type(data_type).cuda()

    if mp_group is not None:
        group_sender = rank // mp_group.size() * mp_group.size()
        torch.distributed.broadcast(inp, group_sender, group=mp_group)
        torch.distributed.broadcast(
            moe.gate.gate.weight.data, group_sender, group=mp_group
        )
        torch.distributed.broadcast(
            moe.gate.gate.bias.data, group_sender, group=mp_group
        )

    inp_raw = inp.clone()
    inp.requires_grad = True

    inp_raw.requires_grad = True
    gate_idx, gate_score = moe.gate(inp_raw)
    moe_out = moe(inp)
    raw_out = moe_raw(inp_raw, gate_idx, gate_score)

    raw_out.mean().backward()
    moe_out.mean().backward()

    return moe_out, raw_out, inp.grad, inp_raw.grad

def _assert_numerical(names, moe_out_list, raw_out_list, rank, precision=1e-3):
    for name, mo, ro in zip(names, moe_out_list, raw_out_list):
        err = (mo - ro).abs().max()
        print("Rank {} {} abs err {}".format(rank, name, err))
        if err > precision:
            sys.stderr.write(f"=========== {name} moe out ==============\n")
            sys.stderr.write("{}\n".format(mo))
            sys.stderr.write(f"=========== {name} raw out ==============\n")
            sys.stderr.write("{}\n".format(ro))
            sys.stderr.write(f"=========== {name} diff ==============\n")
            sys.stderr.write("{}\n{}\n".format((mo - ro).abs(), err))
            assert False

@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("num_expert", [None])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("top_k", [2, 3])
@pytest.mark.parametrize("expert", [ [NaiveExpert for _ in range(4)], [LinearExpert, NaiveExpert, LinearExpert, NaiveExpert, LinearExpert, NaiveExpert, LinearExpert, NaiveExpert] ])
@pytest.mark.parametrize("rank", [0])
@pytest.mark.parametrize("world_size", [1])
@pytest.mark.parametrize("mp_group", [None])
@pytest.mark.parametrize("dp_group", [None])
@pytest.mark.parametrize("world_group", [None])
def test_fmoe_experts(
    batch_size,
    num_expert,
    d_model,
    top_k,
    expert: Union[Type[nn.Module], str],
    rank,
    world_size,
    mp_group,
    dp_group,
    world_group,
):
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    if isinstance(expert, str):
        expert = globals()[expert]

    moe = FMoE(
        num_expert=num_expert,
        d_model=d_model,
        gate=NaiveGate,
        world_size=world_size,
        mp_group=mp_group,
        expert=expert,
        top_k=top_k,
    ).cuda()

    moe_raw = BruteForceMoE(
        expert=expert,
        num_expert=num_expert,
        d_model=d_model,
        world_size=world_size,
        top_k=top_k,
    ).cuda()

    if world_size == 1:
        for expert_moe, expert_raw in zip(moe.experts, moe_raw.experts):
            for para_moe, para_raw in zip(
                expert_moe.parameters(), expert_raw.parameters()
            ):
                para_raw.data = para_moe.data.clone()
    else:
        assert len(moe.experts) >= 1
        for idx, para in enumerate(moe.experts[0].parameters()):
            para_tensor = torch.cat(
                [list(expert.parameters())[idx].unsqueeze(0) for expert in moe.experts]
            )
            para_array = [torch.empty_like(para_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(para_array, para_tensor)
            para_tensor_gathered = torch.cat(para_array, dim=0)
            assert para_tensor_gathered.shape[0] == len(moe_raw.experts)
            for expertID in range(para_tensor_gathered.shape[0]):
                list(moe_raw.experts[expertID].parameters())[
                    idx
                ].data = para_tensor_gathered[expertID]

    moe_out, raw_out, moe_grad_in, raw_grad_in = _perform_forward(
        moe, moe_raw, batch_size, d_model, top_k, rank, mp_group
    )

    def get_experts_grad(experts: List[nn.Module]):
        return torch.stack(
            [
                torch.stack(
                    [
                        p.grad.sum() if p.grad is not None else torch.zeros(1).cuda()
                        for p in item.parameters()
                    ]
                ).sum()
                for item in experts
            ]
        )

    moe_grad, raw_grad = (
        get_experts_grad(moe.experts),
        get_experts_grad(moe_raw.experts),
    )

    if world_size > 1:
        torch.distributed.all_reduce(raw_grad)
        mp_size = mp_group.size() if mp_group else 1
        raw_grad = raw_grad[rank * num_expert : (rank + 1) * num_expert] / mp_size

    moe_out_list = [moe_out, moe_grad, moe_grad_in]
    raw_out_list = [raw_out, raw_grad, raw_grad_in]
    names = ["forward", "backward", "grad_in"]

    _assert_numerical(names, moe_out_list, raw_out_list, rank)