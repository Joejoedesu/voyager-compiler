"""
Schedule hint
"""

from . import loop_enum as le


class Schedule(object):
    """
    schedule_hint: {loop index:[[loop order,loop blocking,loop partitioning @ 1st level mem],[@ 2nd level mem],[3rd .]]}
        loop blocking     ->  temporal loop size
        loop partitioning ->  spatial loop size (spatial unrolling / parallelism)

    partition_loops: the loops which are allowed to be replicated (on top of the defined loop partitioning)
                     to improve HW utilization

    hint_para_index: {mem level: [spatially unrolled loop indexes]}
    """

    def __init__(self, schedule_hint, partition_loops=None, *, order_priority=None):

        self.schedule_hint = schedule_hint
        # Optional inner-to-outer precedence, independent of unit-size loops.
        # Use only where the paired implementation fixes that order.
        self.order_priority = dict(order_priority or {})
        for level, order in self.order_priority.items():
            if level < 0 or sorted(order) != list(range(le.NUM)):
                raise ValueError("order_priority must give every loop exactly once")
            self.order_priority[level] = tuple(order)
        if partition_loops != None:
            self.partition_loops = []
            for l in partition_loops:
                self.partition_loops.append(le.loop_table[l])
        else:
            self.partition_loops = partition_loops

        num_levels = len(list(schedule_hint.values())[0])
        if any(level >= num_levels for level in self.order_priority):
            raise ValueError("order_priority level exceeds the schedule hierarchy")
        hint_para_index = {}
        for loop in schedule_hint:
            for level in range(num_levels):
                if (
                    schedule_hint[loop][level] != None
                    and schedule_hint[loop][level][2] != None
                ):
                    if level not in hint_para_index:
                        hint_para_index[level] = [loop]
                    else:
                        hint_para_index[level].append(loop)
        self.hint_para_index = hint_para_index

    @classmethod
    def schedule(cls, info):
        return cls(info["schedule_hint"], info["partition_loops"])
