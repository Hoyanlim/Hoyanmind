import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

# huggingface的transformers中的预训练模型配置类，包含了模型的各种超参数和配置选项，可以通过继承该类来定义自己的模型配置HoyanmindConfig。
class HoyanmindConfig(PretrainedConfig):
    model_type = "Hoyanmind"

    def __init__(
        self,
        dropout: float = 0.0,              # dropout 概率，防止过拟合
        bos_token_id: int = 1,             # 序列开始 token 的 ID
        eos_token_id: int = 2,             # 序列结束 token 的 ID
        hidden_act: str = "silu",          # 隐藏层激活函数
        hidden_size: int = 512,            # 隐藏层维度
        intermediate_size: int = None,     # FFN 中间层维度，None 时自动计算
        max_position_embeddings: int = 32768,  # 最大位置编码长度
        num_attention_heads: int = 8,      # 注意力头数
        num_hidden_layers: int = 8,        # Transformer 层数
        num_key_value_heads: int = 2,      # KV 注意力头数（GQA）
        vocab_size: int = 6400,            # 词表大小
        rms_norm_eps: float = 1e-05,       # RMSNorm 的稳定性 epsilon
        rope_theta: int = 1000000,         # RoPE 旋转位置编码的 theta 基数
        inference_rope_scaling: bool = False,  # 推理时是否启用 RoPE 缩放
        flash_attention: bool = True,      # 是否启用 FlashAttention 加速
        ############ MoE ############
        use_moe: bool = False,             # 是否启用混合专家（MoE）
        num_experts_per_tok: int = 2,      # 每个 token 激活的专家数
        n_routed_experts: int = 4,         # 路由专家总数
        n_shared_experts: int = 1,         # 共享专家数（始终激活）
        scoring_func: str = "softmax",     # 专家路由评分函数
        aux_loss_alpha: float = 0.01,      # 负载均衡辅助损失权重
        seq_aux: bool = True,              # 是否按序列计算辅助损失
        norm_topk_prob: bool = True,       # 是否对 top-k 专家概率归一化
        **kwargs,
    ):
        super().__init__(**kwargs)  # 调用父类初始化，传递通用配置参数

        self.dropout = dropout                                  # 保存 dropout 概率
        self.bos_token_id = bos_token_id                        # 保存序列开始 token ID
        self.eos_token_id = eos_token_id                        # 保存序列结束 token ID
        self.hidden_act = hidden_act                            # 保存隐藏层激活函数
        self.hidden_size = hidden_size                          # 保存隐藏层维度
        self.intermediate_size = intermediate_size              # 保存 FFN 中间层维度
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置编码长度
        self.num_attention_heads = num_attention_heads          # 保存注意力头数
        self.num_hidden_layers = num_hidden_layers              # 保存 Transformer 层数
        self.num_key_value_heads = num_key_value_heads          # 保存 KV 注意力头数
        self.vocab_size = vocab_size                            # 保存词表大小
        self.rms_norm_eps = rms_norm_eps                        # 保存 RMSNorm epsilon
        self.rope_theta = rope_theta                            # 保存 RoPE theta 基数
        self.inference_rope_scaling = inference_rope_scaling    # 保存是否启用推理 RoPE 缩放
        self.flash_attention = flash_attention                  # 保存是否启用 FlashAttention
        self.use_moe = use_moe                                  # 保存是否启用 MoE
        self.num_experts_per_tok = num_experts_per_tok          # 保存每 token 激活专家数
        self.n_routed_experts = n_routed_experts                # 保存路由专家总数
        self.n_shared_experts = n_shared_experts                # 保存共享专家数
        self.seq_aux = seq_aux                                  # 保存是否按序列计算辅助损失
        self.norm_topk_prob = norm_topk_prob                    # 保存是否归一化 top-k 概率
        self.aux_loss_alpha = aux_loss_alpha                    # 保存辅助损失权重
        self.scoring_func = scoring_func                        # 保存专家路由评分函数

        self.rope_scaling = (   # 启用推理 RoPE 缩放时使用 YaRN 配置，否则为 None
            {
                "beta_fast": 32,                            # 高频部分插值边界
                "beta_slow": 1,                             # 低频部分插值边界
                "factor": 16,                               # 上下文长度扩展倍数
                "original_max_position_embeddings": 2048,  # 原始训练最大长度
                "attention_factor": 1.0,                   # 注意力缩放系数
                "type": "yarn",                            # RoPE 缩放类型：YaRN
            }
            if self.inference_rope_scaling
            else None
        )

#继承nn.Module
class RMSnorm(torch.nn.Module):
    # 初始化
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()  # 调用父类nn.Module的构造函数
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
# _norm
    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * x

#forward
    def forward(self, x):
        return self._norm(x.float()) * self.weight.type_as(x)

def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                    2 * math.log(rope_base)
                )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed
        

    