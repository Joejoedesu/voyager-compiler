"""Default Voyager mapping policy; shared search contains no Voyager hints."""


class VoyagerMappingPolicy:
    search_fully_connected = False
    matrix_only_fusion = False
    speed_only = False

    def __init__(self, config):
        self.config = config

    def schedule(self):
        ic_dim, oc_dim = self.config.pe_array_size
        # L1 IC is outermost; the inner order is pinned to FY > FX > OY > OX.
        # OX/OY innermost is never slower -- the L1 sweep costs
        # ``max(loading, reused_tile) * remaining``, monotone in the reused
        # tile -- and the arrangement of the loops above them ties on both
        # runtime and energy, so FX/FY are fixed to the representative the
        # search's first-seen tie-break picked anyway.  FX=2/FY=3 assumes a
        # square kernel (equal FX/FY blocking); a non-square model may prefer
        # them swapped.
        schedule_constraint = {
            "schedule_hint": {
                "IC": {
                    "level0": {"order": 1, "partitioning_size": ic_dim},
                    "level1": {"order": -1},
                    "level2": {"order": 0},
                    "level3": {"order": 0},
                },
                "OC": {
                    "level0": {"order": 0, "partitioning_size": oc_dim},
                },
                "OX": {
                    "level1": {"order": 0},
                },
                "OY": {
                    "level1": {"order": 1},
                },
                "FX": {
                    "level0": {"blocking_size": 1, "partitioning_size": 1},
                    "level1": {"order": 2},
                    "level2": {"blocking_size": 1, "partitioning_size": 1},
                    "level3": {"blocking_size": 1, "partitioning_size": 1},
                },
                "FY": {
                    "level0": {"blocking_size": 1, "partitioning_size": 1},
                    "level1": {"order": 3},
                    "level2": {"blocking_size": 1, "partitioning_size": 1},
                    "level3": {"blocking_size": 1, "partitioning_size": 1},
                },
            }
        }
        return schedule_constraint

    def prepare(self, size_factory, runtime, actual_batch, **kwargs):
        return size_factory(**kwargs), runtime

    def bufferization(
        self, runtime, architecture, layer, mapping, bank_groups, scratch_slots
    ):
        from voyager_compiler.codegen.transform.bufferize.plan import (
            KernelBufferPlan,
        )

        return KernelBufferPlan(
            bank_groups=bank_groups, scratch_slots=scratch_slots
        )
