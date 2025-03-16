#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:38
# @Desc   : 基于预训练语言模型构建的文本编码器
# --------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Dict, Optional, List, Union, Tuple

class TextEncoder(nn.Module):
    """基于预训练语言模型构建的文本编码器"""
    
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        output_dim: int = 768,
        dropout: float = 0.1,
        freeze_base: bool = False,
        pooling_strategy: str = "cls",
        max_length: int = 512,
        use_layer_weights: bool = False,
        num_hidden_layers: Optional[int] = None
    ):
        """
        初始化文本编码器
        
        Args:
            model_name: 预训练模型名称
            output_dim: 输出维度
            dropout: 丢弃率
            freeze_base: 是否冻结基础模型
            pooling_strategy: 池化策略 ('cls', 'mean', 'max', 'attention', 'weighted')
            max_length: 最大序列长度
            use_layer_weights: 是否使用隐藏层加权组合
            num_hidden_layers: 使用隐藏层数量 (如果为None, 使用所有隐藏层)
        """
        super().__init__()
        
        # 加载预训练模型
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 保存参数
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.use_layer_weights = use_layer_weights
        
        # 如果指定冻结基础模型
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # 加权池化权重
        if use_layer_weights:
            num_layers = num_hidden_layers or self.config.num_hidden_layers
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 注意力池化
        if pooling_strategy == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.Tanh(),
                nn.Linear(self.config.hidden_size, 1, bias=False)
            )
                
        # 输出投影
        if self.config.hidden_size != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.config.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout)
            )
        else:
            self.projection = nn.Identity()
            
    def encode_text(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        直接使用tokenizer编码文本
        
        Args:
            text: 输入文本或文本列表
            
        Returns:
            包含文本嵌入的字典
        """
        # 分词输入
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 移动到模型相同的设备
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 前向传播
        return self.forward(**inputs)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: Token type IDs (optional) 可选
            
        Returns:
            包含:
                - embeddings: 文本嵌入
                - pooled: 池化文本表示
                - hidden_states: 所有隐藏状态 (如果output_hidden_states=True)
        """
        # 获取基础模型输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 获取序列输出
        sequence_output = outputs.last_hidden_state
        
        # 应用不同的池化策略
        if self.pooling_strategy == "cls":
            # 使用[CLS] token表示
            pooled_output = sequence_output[:, 0]
        elif self.pooling_strategy == "mean":
            # 均值池化
            pooled_output = self._mean_pooling(sequence_output, attention_mask)
        elif self.pooling_strategy == "max":
            # 最大池化
            pooled_output = self._max_pooling(sequence_output, attention_mask)
        elif self.pooling_strategy == "attention":
            # 注意力池化
            pooled_output = self._attention_pooling(sequence_output, attention_mask)
        elif self.pooling_strategy == "weighted":
            # 加权层组合
            pooled_output = self._weighted_layer_pooling(outputs.hidden_states, attention_mask)
        else:
            # 默认使用[CLS] token
            pooled_output = sequence_output[:, 0]
        
        # 投影输出
        sequence_output = self.projection(sequence_output)
        pooled_output = self.projection(pooled_output)
        
        return {
            'embeddings': sequence_output,
            'pooled': pooled_output,
            'hidden_states': outputs.hidden_states
        }
    
    def _mean_pooling(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """均值池化"""
        mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
        sum_mask = torch.sum(mask_expanded, 1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask
    
    def _max_pooling(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """最大池化"""
        mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sequence_output = sequence_output.clone()
        sequence_output[mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(sequence_output, 1)[0]
    
    def _attention_pooling(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """注意力池化"""
        # 计算注意力分数
        attention_scores = self.attention_pool(sequence_output).squeeze(-1)
        
        # 掩码填充tokens
        attention_scores = attention_scores.masked_fill(attention_mask.eq(0), -1e9)
        
        # 使用softmax获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 应用注意力权重
        return torch.bmm(attention_weights.unsqueeze(1), sequence_output).squeeze(1)
    
    def _weighted_layer_pooling(self, hidden_states: Tuple[torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        """加权组合隐藏层"""
        # 归一化层权重
        norm_weights = F.softmax(self.layer_weights, dim=0)
        
        # 跳过第一个元素，它是嵌入层输出
        hidden_states = hidden_states[1:]
        
        # 加权求和隐藏状态
        weighted_sum = torch.zeros_like(hidden_states[0][:, 0])
        for i, hidden_state in enumerate(hidden_states):
            weighted_sum += norm_weights[i] * hidden_state[:, 0]
            
        return weighted_sum
        
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.config.hidden_size
    
    def get_output_dim(self) -> int:
        """获取投影后的输出维度"""
        if isinstance(self.projection, nn.Identity):
            return self.config.hidden_size
        else:
            return self.projection[0].out_features 