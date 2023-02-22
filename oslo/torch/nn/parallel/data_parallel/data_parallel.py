from typing import List, Optional
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel.utils import add_wrapper

from .distributed_data_parallel import _DistributedDataParallel


def DataParallel(
    module,
    optimizer,
    parallel_context: ParallelContext,
    zero_stage: int = 0,
    # transformer_wrap_layers: Optional[List] = None,
    # mixed_precision: Optional[MixedPrecision] = None,
    # cpu_offload: bool = False,
):
    if zero_stage == 0:
        return (
            DistributedDataParallel(module, parallel_context=parallel_context),
            optimizer,
        )

    # elif zero_stage == 1:
    #     module = FullyShardedDataParallel(
    #         module=module,
    #         parallel_context=parallel_context,
    #         sharding_strategy=ShardingStrategy.NO_SHARD,
    #         transformer_wrap_layers=transformer_wrap_layers,
    #         mixed_precision=mixed_precision,
    #         cpu_offload=CPUOffload(offload_params=cpu_offload),
    #     )
    #     optimizer = ZeroRedundancyOptimizer(
    #         module.parameters(),
    #         optimizer_class=optimizer.__class__,
    #         **optimizer.defaults,
    #     )
    #     return module, optimizer

    # elif zero_stage == 2:
    #     module = FullyShardedDataParallel(
    #         module=module,
    #         parallel_context=parallel_context,
    #         sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    #         transformer_wrap_layers=transformer_wrap_layers,
    #         mixed_precision=mixed_precision,
    #         cpu_offload=CPUOffload(offload_params=cpu_offload),
    #     )
    #     optimizer = ZeroRedundancyOptimizer(
    #         module.parameters(),
    #         optimizer_class=optimizer.__class__,
    #         **optimizer.defaults,
    #     )
    #     return module, optimizer

    # elif zero_stage == 3:
    #     module = FullyShardedDataParallel(
    #         module=module,
    #         parallel_context=parallel_context,
    #         sharding_strategy=ShardingStrategy.FULL_SHARD,
    #         transformer_wrap_layers=transformer_wrap_layers,
    #         mixed_precision=mixed_precision,
    #         cpu_offload=CPUOffload(offload_params=cpu_offload),
    #     )
    #     optimizer = ZeroRedundancyOptimizer(
    #         module.parameters(),
    #         optimizer_class=optimizer.__class__,
    #         **optimizer.defaults,
    #     )
    #     return module, optimizer
    else:
        raise ValueError("param `zero_stage` must be one of the 0, 1, 2, 3.")


def DistributedDataParallel(
    module, parallel_context, bucket_cap_mb=25, rebuild_bucket=True
):
    ddp = _DistributedDataParallel(
        module,
        parallel_context,
        bucket_cap_mb=bucket_cap_mb,
        rebuild_bucket=rebuild_bucket,
    )
    add_wrapper(
        module, mode=ParallelMode.DATA, wrapper=ddp, parallel_context=parallel_context
    )
    setattr(module, "forward", ddp.forward)
    return module
