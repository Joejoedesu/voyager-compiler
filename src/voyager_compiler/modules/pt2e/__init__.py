from transformers.models import phi3

from voyager_compiler.modules.pt2e.modeling_phi3 import Phi3RotaryEmbedding

phi3.modeling_phi3.Phi3RotaryEmbedding = Phi3RotaryEmbedding
