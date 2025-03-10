"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from .transformer import Transformer
from .utils import load_pretrained_weights, as_tuple
from .configs import PRETRAINED_MODELS


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self,
        name: Optional[str] = None,
        pretrained: bool = False,
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3,
        image_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        resize_positional_embedding=None
    ):
        super().__init__()

        # Configuration
        if name is None:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            assert not resize_positional_embedding, check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size
        self.dim = dim

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1

        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()

        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads,
                                       ff_dim=ff_dim, dropout=dropout_rate)

        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)

        # Initialize weights
        self.init_weights()

        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            load_pretrained_weights(
                self, name,
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(
                    image_size != pretrained_image_size),
            )

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    # nn.init.constant(m.bias, 0)
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape  # 64, 3, 224, 224
        x = self.patch_embedding(x)  # 64, 768, 7, 7
        x = x.flatten(2).transpose(1, 2)  # 64, 49, 768
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(
                b, -1, -1), x), dim=1)  # 64, 50, 768
        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)  # 添加位置编码 (64, 50, 768)
        x = self.transformer(x)  # (64, 50, 768)
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]  # b,d
            return_feature = x
            x = self.fc(x)  # b,num_classes
        return x, return_feature


class WeatherModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim)  # word2vec  (天气的数目, 映射之后的向量维度, )

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, out_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return_feature = x

        x = self.classifier(x)
        return x, return_feature
        # return x, x


class MultiModalModel(nn.Module):
    def __init__(
        self,
        num_embeddings,
        weather_embedding_dim,
        layer_num,
        weather_out_dim,
        name: Optional[str] = None,
        pretrained: bool = False,
        num_classes=11,
    ):
        super().__init__()
        # 保存参数
        self.num_embeddings = num_embeddings
        self.weather_embedding_dim = weather_embedding_dim
        self.num_classes = num_classes
        self.weather_out_dim = weather_out_dim

        # 图像模型
        self.image_model = ViT(
            name=name, pretrained=pretrained, num_classes=num_classes)
        # 选择层数
        self.image_model.transformer.blocks = self.image_model.transformer.blocks[:layer_num]
        vit_dim = self.image_model.dim

        # 天气模型
        self.weather_model = WeatherModel(
            num_embeddings=num_embeddings,
            embedding_dim=weather_embedding_dim,
            num_classes=num_classes,
            out_dim=weather_out_dim)

        final_dim = vit_dim+weather_out_dim
        # 融合图像特征和天气特征之后的分类层
        self.fc = nn.Sequential(
            nn.Linear(final_dim, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, weather):
        # 提取图像特征
        pred1, img = self.image_model(img)

        # 提取天气特征
        pred2, weather = self.weather_model(weather)

        # 拼接图像和天气特征
        x = torch.cat((img, weather), dim=1)

        # 用拼接之后的特征做分类
        pred3 = self.fc(x)
        return pred1, pred2, pred3
