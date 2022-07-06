import torch.nn as nn
import torchvision.models as models


def alexnet_customize(out, pretrained=True):
    """ 自定义 ResNet 最后一层的输出数目 """

    net = models.alexnet(pretrained)
    num_features = net.classifier[6].in_features
    features = list(net.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, out)])
    net.classifier = nn.Sequential(*features)

    return net


def inception_customize(out, layer=34, pretrained=True):
    """ 自定义 ResNet 最后一层的输出数目 """

    net = models.inception_v3(pretrained)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, out)

    return net


def resnet_custom(out, layer=34, pretrained=True):
    """ 自定义 ResNet 最后一层的输出数目 """
    if 34 == layer:
        net = models.resnet34(pretrained)
    elif 101 == layer:
        net = models.resnet101(pretrained)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, out)

    return net


def vgg16_customize(out, pretrained=True):
    """替换 vgg 网络最后一层的输出向量

    Params:
        out: 输出向量个数
    """
    net = models.vgg16_bn(pretrained)
    num_features = net.classifier[6].in_features
    features = list(net.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, out)])
    net.classifier = nn.Sequential(*features)  # Replace the model classifier

    return net


# def get_nets(model_config):
#     model_name = model_config['g_name']
#     out_features = model_config['out_features']
#
#     if 'resnet34' == model_name:
#         model = resnet_custom(out_features, 34)
#     elif 'resnet101' == model_name:
#         model = resnet_custom(out_features, 101)
#     elif 'vgg16' == model_name:
#         model = vgg16_customize(out_features)
#
#     print(model)
#     return model


def get_nets(model_name, out_features):

    if 'resnet34' == model_name:
        model = resnet_custom(out_features, 34)
    elif 'resnet101' == model_name:
        model = resnet_custom(out_features, 101)
    elif 'vgg16' == model_name:
        model = vgg16_customize(out_features)
    elif 'inception' == model_name:
        model = inception_customize(out_features)
    elif 'alexnet' == model_name:
        model = alexnet_customize(out_features)

    print(model)
    return model