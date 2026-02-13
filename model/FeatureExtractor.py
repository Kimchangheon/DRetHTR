import torch
import torch.nn as nn
from torchvision import models
from einops.layers.torch import Rearrange
from einops import repeat

class FeatureExtractor(nn.Module):
    def __init__(self, feature_extractor, image_size=None, patch_size=None, num_channels=None,
                 dim=None, patch_order=None, cnn_dropout=None, img_emb_dropout=None):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = feature_extractor
        self.img_emb_dropout = nn.Dropout(img_emb_dropout)
        ocs = 2 # output channel scale factor for shallow cnn

        if feature_extractor == "Patch_embedding":
            image_height, image_width = image_size
            patch_height, patch_width = patch_size

            assert image_height % patch_height == 0 and image_width % patch_width == 0, (
                f"Image dimensions must be divisible by the patch size. "
                f"Given: image_height={image_height}, image_width={image_width}, "
                f"patch_height={patch_height}, patch_width={patch_width}"
            )
            num_patches = (image_height // patch_height) * (image_width // patch_width)
            patch_dim = num_channels * patch_height * patch_width

            if patch_order == 0:
                self.to_patch_embedding = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (w h) (p1 p2 c)', p1=patch_height, p2=patch_width),
                    nn.LayerNorm(patch_dim),
                    nn.Linear(patch_dim, dim),
                    nn.LayerNorm(dim),
                )
            elif patch_order == 1:
                self.to_patch_embedding = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                    nn.LayerNorm(patch_dim),
                    nn.Linear(patch_dim, dim),
                    nn.LayerNorm(dim),
                )
            else:
                raise ValueError("Invalid value for patch order.")

            self.img_pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        elif feature_extractor == "Shallow_CNN":
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=ocs * 8, kernel_size=(6, 4), stride=(2, 2))
            self.leaky_relu1 = nn.LeakyReLU()
            self.dropout1 = nn.Dropout(p=cnn_dropout)
            self.conv2 = nn.Conv2d(in_channels=ocs * 8, out_channels=ocs * 32, kernel_size=(6, 4), stride=(1, 1))
            self.leaky_relu2 = nn.LeakyReLU()
            self.dropout2 = nn.Dropout(p=cnn_dropout)
            self.max_pool1 = nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2))
            self.conv3 = nn.Conv2d(in_channels=ocs * 32, out_channels=ocs * 64, kernel_size=(3, 3), stride=(1, 1))
            self.leaky_relu3 = nn.LeakyReLU()
            self.dropout3 = nn.Dropout(p=cnn_dropout)
            self.max_pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            in_channels = ocs * 64
            reduced_channels = ocs * 64
            height = 4
            self.linear = nn.Linear(in_channels * height, reduced_channels)
            self.img_pos_embedding = nn.Parameter(torch.randn(1, 276, dim))

        elif feature_extractor == "Shallow_CNN_512":
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=ocs * 8, kernel_size=(6, 4), stride=(2, 2))
            self.leaky_relu1 = nn.LeakyReLU()
            self.dropout1 = nn.Dropout(p=cnn_dropout)
            self.conv2 = nn.Conv2d(in_channels=ocs * 8, out_channels=ocs * 32, kernel_size=(6, 4), stride=(1, 1))
            self.leaky_relu2 = nn.LeakyReLU()
            self.dropout2 = nn.Dropout(p=cnn_dropout)
            self.max_pool1 = nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2))
            self.conv3 = nn.Conv2d(in_channels=ocs * 32, out_channels=ocs * 64, kernel_size=(3, 3), stride=(1, 1))
            self.leaky_relu3 = nn.LeakyReLU()
            self.dropout3 = nn.Dropout(p=cnn_dropout)
            self.max_pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            self.img_pos_embedding = nn.Parameter(torch.randn(1, 276, dim))

        elif feature_extractor == "Shallow_CNN_1280":
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(6, 4), stride=(2, 2))
            self.leaky_relu1 = nn.LeakyReLU()
            self.dropout1 = nn.Dropout(p=cnn_dropout)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(6, 4), stride=(1, 1))
            self.leaky_relu2 = nn.LeakyReLU()
            self.dropout2 = nn.Dropout(p=cnn_dropout)
            self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1))
            self.leaky_relu3 = nn.LeakyReLU()
            self.dropout3 = nn.Dropout(p=cnn_dropout)
            self.max_pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            self.img_pos_embedding = nn.Parameter(torch.randn(1, 276, dim))

        elif "ResNet50" in feature_extractor:
            resnet = models.resnet50(pretrained=True)

            if num_channels != 3:
                resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            if "ResNet50_simple" in feature_extractor:
                resnet.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False)
                resnet.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 1), bias=False)
            else:
                resnet.layer2[0].conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                resnet.layer2[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.features = self.add_dropout_to_relu(self.features, cnn_dropout)
            self.img_pos_embedding = nn.Parameter(torch.randn(1, 140, dim))

        elif "efficientnet_v2_s" == feature_extractor:
            try:
                # Try loading with specified weights
                model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
                print("Model loaded successfully with IMAGENET1K_V1 weights.")
            except Exception as e:
                print(f"An error occurred: {e}")
            model.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False)
            self.features = nn.Sequential(*list(model.children())[:-2])
            self.features = self.add_dropout_to_relu(self.features, cnn_dropout)
            self.linear = nn.Linear(1280 * 2, dim)
            self.img_pos_embedding = nn.Parameter(torch.randn(1, 140, dim))

        elif "efficientnet_v2_m" == feature_extractor:
            try:
                # Try loading with specified weights
                model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
                print("Model loaded successfully with IMAGENET1K_V1 weights.")
            except Exception as e:
                print(f"An error occurred: {e}")
            model.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False)
            self.features = nn.Sequential(*list(model.children())[:-2])
            self.features = self.add_dropout_to_relu(self.features, cnn_dropout)
            self.linear = nn.Linear(1280 * 2, dim)
            self.img_pos_embedding = nn.Parameter(torch.randn(1, 140, dim))

        elif "efficientnet_v2_l" == feature_extractor:
            try:
                # Try loading with specified weights
                model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
                print("Model loaded successfully with IMAGENET1K_V1 weights.")
            except Exception as e:
                print(f"An error occurred: {e}")
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False)
            self.features = nn.Sequential(*list(model.children())[:-2])
            self.features = self.add_dropout_to_relu(self.features, cnn_dropout)
            self.linear = nn.Linear(1280 * 2, dim)
            self.img_pos_embedding = nn.Parameter(torch.randn(1, 140, dim))

        elif "regnet_y_128gf" == feature_extractor:
            model = models.regnet_y_128gf(weights=models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1)
            model.stem[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False)
            self.features = nn.Sequential(*list(model.children())[:-2])
            self.features = self.add_dropout_to_relu(self.features, cnn_dropout)
            self.linear = nn.Linear(7392 * 2, dim)
            self.img_pos_embedding = nn.Parameter(torch.randn(1, 140, dim))

    def add_dropout_to_relu(self, module, dropout):
        for name, child in module.named_children():
            if isinstance(child, (nn.ReLU, nn.SiLU, nn.LeakyReLU)):  # Check for ReLU or SiLU
                module.add_module(name, nn.Sequential(child, nn.Dropout(p=dropout)))
            elif isinstance(child, nn.Sequential) or isinstance(child, nn.ModuleList):
                self.add_dropout_to_relu(child, dropout)
            else:
                self.add_dropout_to_relu(child, dropout)
        return module

    def forward(self, img):
        if self.feature_extractor == "Patch_embedding":
            x = self.to_patch_embedding(img)
            b, n, _ = x.shape
            image_patches_length = n
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            image_patches_length += 1
            x += self.img_pos_embedding[:, :(n + 1)]

        elif "Shallow_CNN" in self.feature_extractor:
            x = self.conv1(img)
            x = self.leaky_relu1(x)
            x = self.dropout1(x)

            x = self.conv2(x)
            x = self.leaky_relu2(x)
            x = self.dropout2(x)

            x = self.max_pool1(x)

            x = self.conv3(x)
            x = self.leaky_relu3(x)
            x = self.dropout3(x)

            x = self.max_pool2(x)

            b, c, h, w = x.shape
            x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
            if self.feature_extractor == "Shallow_CNN":
                x = self.linear(x)

            b, n, _ = x.shape #    image_tokens_length = n
            x += self.img_pos_embedding[:, :n]

        elif "ResNet50" in self.feature_extractor:
            x = self.features(img)
            b, c, h, w = x.shape
            x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
            if "ResNet50_simple" in self.feature_extractor:
                x = self.linear(x)
            else:
                x = self.linear(x)
                x = self.linear2(x)

            b, n, _ = x.shape
            x += self.img_pos_embedding[:, :n]

        elif self.feature_extractor in ["efficientnet_v2_s","efficientnet_v2_m", "efficientnet_v2_l", "regnet_y_128gf"]:
            x = self.features(img)
            b, c, h, w = x.shape
            x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
            x = self.linear(x)
            b, n, _ = x.shape
            x += self.img_pos_embedding[:, :n]

        x = self.img_emb_dropout(x)
        return x
