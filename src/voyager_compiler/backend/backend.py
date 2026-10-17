"""What a target adds to the shared ``compile()``: a ``Backend``.

A config's ``backend()`` returns it.  ``compile()`` bufferizes with its
builders; the rest of the pipeline is the same for every target.
"""

from dataclasses import dataclass

from voyager_compiler.codegen.transform.bufferize import BuilderRule


@dataclass(frozen=True)
class Backend:
    """A target's hooks into ``compile()``.

    Attributes:
        builders: The rules that pick each compute node's builder, tried in
            order; a rule ahead of ``SHARED_BUILDERS`` replaces the shared
            builder for the nodes it matches.
    """

    builders: tuple[BuilderRule, ...]
