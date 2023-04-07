import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from peach.enc_utils.sim_metric import Similarity
from peach.enc_utils.general import preproc_inputs

from transformers.modeling_utils import PreTrainedModel

# from peach.nn_utils.optimal_trans import ipot, trace, optimal_transport_dist
# from peach.nn_utils.general import mask_out_cls_sep


class FLOPS(nn.Module):
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)


class LearnerMixin(PreTrainedModel):
    def __init__(self, model_args, config, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

    @property
    def encoding_doc(self):
        return self.encoder

    @property
    def encoding_query(self):
        if self.query_encoder is None:
            return self.encoder
        return self.query_encoder

    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor(self, target_tensor):
        if dist.is_initialized() and dist.get_world_size() > 1 and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor
            target_tensor_gathered = torch.cat(target_tensor_list, 0)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1
