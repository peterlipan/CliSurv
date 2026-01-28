import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, n_layers=2, dropout=0.0, activation='relu'):
        super(MLP, self).__init__()

        self.activation = self.get_activation_fn(activation)

        layers = []
        for i in range(n_layers):
            in_dim = d_in if i == 0 else d_hid
            out_dim = d_hid if i < n_layers - 1 else d_out
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(self.activation)
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) 
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

    @staticmethod
    def get_activation_fn(name):
        if name == "relu":
            return nn.ReLU()
        elif name == "elu":
            return nn.ELU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.1)
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")


class VisionTransformer(nn.Module):
    def __init__(self, n_features, patch_size, pretrained=True):
        super(VisionTransformer, self).__init__()
        config = ViTConfig.from_pretrained(pretrained) if pretrained else ViTConfig()
        config.image_size = n_features
        config.patch_size = patch_size
        self.config = config
        self.d_hid = config.hidden_size
        self.vit = ViTModel(config, add_pooling_layer=False, use_cls_token=False)
        if pretrained:
            self.vit = ViTModel.from_pretrained(pretrained, config=config, add_pooling_layer=False, use_cls_token=False, ignore_mismatched_sizes=True)
        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        outputs = self.vit(x)
        sequence_output = outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output.transpose(1, 2)).squeeze(-1)
        return pooled_output


class TorchVisionModels(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(TorchVisionModels, self).__init__()
        model = getattr(torchvision.models, model_name)(pretrained=pretrained)
        if model_name.startswith('resnet'):
            self.d_hid = model.fc.in_features
            model.fc = nn.Identity()
        
        elif model_name.startswith('densenet'):
            self.d_hid = model.classifier.in_features
            model.classifier = nn.Identity()
        
        elif model_name.startswith('efficientnet'):
            self.d_hid = model.classifier[1].in_features
            model.classifier[1] = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model = model

    def forward(self, x):
        return self.model(x)


def get_encoder(args):
    if args.backbone.lower() == 'mlp':
        return MLP(d_in=args.n_features, d_hid=args.d_hid, d_out=args.d_hid, n_layers=args.n_layers, dropout=args.dropout, activation=args.activation)
    elif args.backbone.lower() == 'vit':
        pretrained = "WinKawaks/vit-tiny-patch16-224" if args.pretrained else ""
        return VisionTransformer(n_features=args.n_features, patch_size=args.patch_size, pretrained=pretrained)
    else:
        return TorchVisionModels(model_name=args.backbone, pretrained=args.pretrained)