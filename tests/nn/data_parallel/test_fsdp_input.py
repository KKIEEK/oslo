# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with different input types. """

import os
import tempfile

import pytest
import torch
from torch.nn import Linear, Module
from torch.optim import SGD

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel.distributed import FullyShardedDataParallel as FSDP
from oslo.torch.nn.parallel.distributed import TrainingState
from oslo.torch.utils import torch_version
from oslo.torch.utils.testing import dist_init, rmf, skip_if_no_cuda, teardown


# A fixture to get tempfiles and ensure they are cleaned up.
@pytest.fixture()
def temp_files():
    num = 2  # dist_init needs 2 files
    files = [tempfile.mkstemp()[1] for _ in range(num)]

    yield tuple(files)

    # temp files could have been removed, so we use rmf.
    for name in files:
        rmf(name)


# We only test on GPU since mix-precision only works on GPU.
@skip_if_no_cuda
@pytest.mark.parametrize(
    "fsdp_config",
    [{}, {"mixed_precision": True}],
)
@pytest.mark.parametrize("input_cls", [dict, list])
def test_input_type(temp_files, fsdp_config, input_cls):
    """Test FSDP with input being a list or a dict, only single GPU."""

    if torch_version() < (1, 7, 0):
        # This test runs multiple test cases in a single process. On 1.6.0 it
        # throw an error like this:
        #     RuntimeError: Container is already initialized! Cannot initialize it twice!
        pytest.skip("older pytorch doesn't work well with single process dist_init multiple times")

    os.environ["RANK"] = str(0)
    os.environ["LOCAL_RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(29500)

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=1
    )
    assert parallel_context, "Dist init failed"

    assert isinstance(fsdp_config, dict), str(fsdp_config)

    class Model(Module):
        def __init__(self):
            super().__init__()
            self.layer = Linear(4, 4)

        def forward(self, input):
            if isinstance(input, list):
                input = input[0]
            else:
                assert isinstance(input, dict), input
                input = input["in"]
            return self.layer(input)

    model = FSDP(Model(), parallel_context, **fsdp_config).cuda()
    optim = SGD(model.parameters(), lr=0.1)

    for _ in range(5):
        in_data = torch.rand(64, 4).cuda()
        in_data.requires_grad = True
        if input_cls is list:
            in_data = [in_data]
        else:
            assert input_cls is dict
            in_data = {"in": in_data}

        out = model(in_data)
        out.sum().backward()
        optim.step()
        optim.zero_grad()

    model.assert_state(TrainingState.IDLE)

    teardown()
