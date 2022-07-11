import torch
from torch import nn
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    '''
    for example.
    :param v: 63
    :param divisor: 8
    :param min_value: equals divisor, 8
    :return:
    '''
    if min_value is None:
        min_value = divisor
    # max(8, int(63 + 4) // 8 * 8)) = max(8, 7 * 8) = 56
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # 56 < 0.9 * 63
    if new_v < 0.9 * v:
        # 56 + 8 = 64
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        # round: 5舍6入
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),

            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 2048

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                # t: channel_expand_ratio
                # c: channels
                # n: nums of layer
                # s: stride

                # (1, 32, 256, 256) -> (1, 16, 256, 256)
                [1, 16, 1, 1],

                # (1, 16, 256, 256) -> (1, 24, 128, 128)
                [6, 24, 2, 2],

                # (1, 24, 128, 128) -> (1, 32, 64, 64)
                [6, 32, 3, 2],

                # (1, 32, 64, 64) -> (1, 64, 32, 32)
                [6, 64, 4, 2],

                # (1, 64, 32, 32) -> (1, 96, 32, 32)
                [6, 96, 3, 1],

                # (1, 96, 32, 32) -> (1, 160, 16, 16)
                [6, 160, 3, 2],

                # (1, 160, 16, 16) -> (1, 320, 16, 16)
                [6, 320, 1, 1],

                # ------------------------------------#
                #            自定义的卷积层            #
                # ------------------------------------#

                # (1, 320, 16, 16) -> (1, 1024, 16, 16)
                [6, 1024, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        # (1, 3, 512, 512) -> (1, 32, 256, 256)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        # 这里循环将卷积层头对尾连接起来
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # (1, 32, 256, 256) -> (1, last_channel, 16, 16)
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # (1, last_channel, 16, 16) -> (1, last_channel)
        # (1, last_channel) -> (1, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)  # torch.Size([1, last_channel, 16, 16])
        # print(x.shape)

        # torch.mean()返回的是一个标量，即求均值之后默认删掉这个维度
        # x = x.mean([2, 3])  # torch.Size([1, last_channel])
        # x = self.classifier(x)  # torch.Size([1, num_classes])
        return x


def mobilenet_v2(pretrained=False, progress=True):
    model = MobileNetV2()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=progress)  # model_dir="model_data"

        # 只去掉最后的全连接层，其他层完全不同，用这个
        # weights_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].numel() == v.numel()}

        # 如果除了去掉最后的全连接层，中间自己还添加了卷积层，用这个
        weights_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)

        print('----- mobilenetv2 weights pretrained -----')
        print('missing_keys:', missing_keys)
        print('unexpected_keys:', unexpected_keys)

        for param in model.features.parameters():
            param.requires_grad = False

        # optimizer中的参数
        # params = [p for p in model.parameters() if p.requires_grad]
    return model


if __name__ == "__main__":
    input = torch.randn(1, 3, 512, 512)
    model = mobilenet_v2(pretrained=True)
    output = model(input)
    print(output.shape)
