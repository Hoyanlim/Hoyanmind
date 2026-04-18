from transformers import PretrainedConfig
import torch  # noqa: E402
import torch.nn as nn

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