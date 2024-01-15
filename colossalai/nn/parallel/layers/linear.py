from .colo_module import ColoModule
from colossalai.tensor import ComputePattern, distspec, ProcessGroup, ShardSpec, ReplicaSpec


class ColoLinear(ColoModule):

    def __init__(self, model_name):
        super(ColoLinear, self).__init__()
        self.model_name = model_name
        if 'llama' in self.model_name:
            self._register_shard_params(['weight']) # LLaMA: bias=False
        elif 'opt' in self.model_name:
            self._register_shard_params(['weight', 'bias'])
        

    def register(self, compute_pattern, pg: ProcessGroup):
        if not compute_pattern in self._allowed_patterns:
            if ComputePattern.TP1D == compute_pattern:
                self._set_TP1D(pg)

    def _set_TP1D(self, pg):
        # TP1D Row Linear
        _compute_pattern = ComputePattern.TP1D
        if 'llama' in self.model_name:
            self._register_allowed_patterns(
                compute_pattern=_compute_pattern,
                dist_specs={'weight': ShardSpec([-1], [pg.tp_world_size()])},  
                mode='row',
            )
        elif 'opt' in self.model_name:
            self._register_allowed_patterns(
                compute_pattern=_compute_pattern,
                dist_specs={
                    'weight': ShardSpec([-1], [pg.tp_world_size()]),
                    'bias': None
                },
                mode='row',
            )

        # TP1D Col Linear
        if 'llama' in self.model_name:
            self._register_allowed_patterns(
                compute_pattern=_compute_pattern,
                dist_specs={'weight': ShardSpec([0], [pg.tp_world_size()]),},  ## default was 0
                mode='col',
            )
        elif 'opt' in self.model_name:
                self._register_allowed_patterns(
                compute_pattern=_compute_pattern,
                dist_specs={
                    'weight': ShardSpec([0], [pg.tp_world_size()]),
                    'bias': ReplicaSpec()
                },
                mode='col',
            )

        self._set_default(compute_pattern=_compute_pattern, target_mode='row') # target_mode='col'
