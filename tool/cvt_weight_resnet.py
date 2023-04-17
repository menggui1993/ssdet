import torch

def load_conv(src_dict, dst_dict, src_node, dst_node):
    dst_dict[dst_node+'.weight'] = src_dict[src_node+'.weight']
    if src_node+'.bias' in src_dict:
        dst_dict[dst_node+'.bias'] = src_dict[src_node+'.bias']

def load_bn(src_dict, dst_dict, src_node, dst_node):
    dst_dict[dst_node+'.running_mean'] = src_dict[src_node+'.running_mean']
    dst_dict[dst_node+'.running_var'] = src_dict[src_node+'.running_var']
    dst_dict[dst_node+'.weight'] = src_dict[src_node+'.weight']
    dst_dict[dst_node+'.bias'] = src_dict[src_node+'.bias']


state_dict = torch.load("/home/twc/.cache/torch/checkpoints/resnet50-19c8e357.pth")
print(state_dict.keys())

ss_state_dict = {}

# conv1
load_conv(state_dict, ss_state_dict, 'conv1', 'conv1.conv')
load_bn(state_dict, ss_state_dict, 'bn1', 'conv1.norm')

# layer1
load_conv(state_dict, ss_state_dict, 'layer1.0.conv1', 'stage_1.res_block_0.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer1.0.bn1', 'stage_1.res_block_0.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer1.0.conv2', 'stage_1.res_block_0.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer1.0.bn2', 'stage_1.res_block_0.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer1.0.conv3', 'stage_1.res_block_0.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer1.0.bn3', 'stage_1.res_block_0.conv3.norm')
load_conv(state_dict, ss_state_dict, 'layer1.0.downsample.0', 'stage_1.res_block_0.downsample.conv')
load_bn(state_dict, ss_state_dict, 'layer1.0.downsample.1', 'stage_1.res_block_0.downsample.norm')

load_conv(state_dict, ss_state_dict, 'layer1.1.conv1', 'stage_1.res_block_1.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer1.1.bn1', 'stage_1.res_block_1.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer1.1.conv2', 'stage_1.res_block_1.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer1.1.bn2', 'stage_1.res_block_1.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer1.1.conv3', 'stage_1.res_block_1.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer1.1.bn3', 'stage_1.res_block_1.conv3.norm')

load_conv(state_dict, ss_state_dict, 'layer1.2.conv1', 'stage_1.res_block_2.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer1.2.bn1', 'stage_1.res_block_2.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer1.2.conv2', 'stage_1.res_block_2.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer1.2.bn2', 'stage_1.res_block_2.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer1.2.conv3', 'stage_1.res_block_2.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer1.2.bn3', 'stage_1.res_block_2.conv3.norm')

# layer2
load_conv(state_dict, ss_state_dict, 'layer2.0.conv1', 'stage_2.res_block_0.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer2.0.bn1', 'stage_2.res_block_0.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer2.0.conv2', 'stage_2.res_block_0.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer2.0.bn2', 'stage_2.res_block_0.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer2.0.conv3', 'stage_2.res_block_0.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer2.0.bn3', 'stage_2.res_block_0.conv3.norm')
load_conv(state_dict, ss_state_dict, 'layer2.0.downsample.0', 'stage_2.res_block_0.downsample.conv')
load_bn(state_dict, ss_state_dict, 'layer2.0.downsample.1', 'stage_2.res_block_0.downsample.norm')

load_conv(state_dict, ss_state_dict, 'layer2.1.conv1', 'stage_2.res_block_1.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer2.1.bn1', 'stage_2.res_block_1.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer2.1.conv2', 'stage_2.res_block_1.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer2.1.bn2', 'stage_2.res_block_1.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer2.1.conv3', 'stage_2.res_block_1.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer2.1.bn3', 'stage_2.res_block_1.conv3.norm')

load_conv(state_dict, ss_state_dict, 'layer2.2.conv1', 'stage_2.res_block_2.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer2.2.bn1', 'stage_2.res_block_2.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer2.2.conv2', 'stage_2.res_block_2.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer2.2.bn2', 'stage_2.res_block_2.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer2.2.conv3', 'stage_2.res_block_2.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer2.2.bn3', 'stage_2.res_block_2.conv3.norm')

load_conv(state_dict, ss_state_dict, 'layer2.3.conv1', 'stage_2.res_block_3.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer2.3.bn1', 'stage_2.res_block_3.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer2.3.conv2', 'stage_2.res_block_3.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer2.3.bn2', 'stage_2.res_block_3.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer2.3.conv3', 'stage_2.res_block_3.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer2.3.bn3', 'stage_2.res_block_3.conv3.norm')

# layer3
load_conv(state_dict, ss_state_dict, 'layer3.0.conv1', 'stage_3.res_block_0.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer3.0.bn1', 'stage_3.res_block_0.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer3.0.conv2', 'stage_3.res_block_0.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer3.0.bn2', 'stage_3.res_block_0.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer3.0.conv3', 'stage_3.res_block_0.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer3.0.bn3', 'stage_3.res_block_0.conv3.norm')
load_conv(state_dict, ss_state_dict, 'layer3.0.downsample.0', 'stage_3.res_block_0.downsample.conv')
load_bn(state_dict, ss_state_dict, 'layer3.0.downsample.1', 'stage_3.res_block_0.downsample.norm')

load_conv(state_dict, ss_state_dict, 'layer3.1.conv1', 'stage_3.res_block_1.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer3.1.bn1', 'stage_3.res_block_1.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer3.1.conv2', 'stage_3.res_block_1.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer3.1.bn2', 'stage_3.res_block_1.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer3.1.conv3', 'stage_3.res_block_1.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer3.1.bn3', 'stage_3.res_block_1.conv3.norm')

load_conv(state_dict, ss_state_dict, 'layer3.2.conv1', 'stage_3.res_block_2.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer3.2.bn1', 'stage_3.res_block_2.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer3.2.conv2', 'stage_3.res_block_2.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer3.2.bn2', 'stage_3.res_block_2.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer3.2.conv3', 'stage_3.res_block_2.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer3.2.bn3', 'stage_3.res_block_2.conv3.norm')

load_conv(state_dict, ss_state_dict, 'layer3.3.conv1', 'stage_3.res_block_3.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer3.3.bn1', 'stage_3.res_block_3.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer3.3.conv2', 'stage_3.res_block_3.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer3.3.bn2', 'stage_3.res_block_3.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer3.3.conv3', 'stage_3.res_block_3.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer3.3.bn3', 'stage_3.res_block_3.conv3.norm')

load_conv(state_dict, ss_state_dict, 'layer3.4.conv1', 'stage_3.res_block_4.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer3.4.bn1', 'stage_3.res_block_4.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer3.4.conv2', 'stage_3.res_block_4.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer3.4.bn2', 'stage_3.res_block_4.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer3.4.conv3', 'stage_3.res_block_4.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer3.4.bn3', 'stage_3.res_block_4.conv3.norm')

load_conv(state_dict, ss_state_dict, 'layer3.5.conv1', 'stage_3.res_block_5.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer3.5.bn1', 'stage_3.res_block_5.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer3.5.conv2', 'stage_3.res_block_5.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer3.5.bn2', 'stage_3.res_block_5.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer3.5.conv3', 'stage_3.res_block_5.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer3.5.bn3', 'stage_3.res_block_5.conv3.norm')

# layer4
load_conv(state_dict, ss_state_dict, 'layer4.0.conv1', 'stage_4.res_block_0.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer4.0.bn1', 'stage_4.res_block_0.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer4.0.conv2', 'stage_4.res_block_0.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer4.0.bn2', 'stage_4.res_block_0.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer4.0.conv3', 'stage_4.res_block_0.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer4.0.bn3', 'stage_4.res_block_0.conv3.norm')
load_conv(state_dict, ss_state_dict, 'layer4.0.downsample.0', 'stage_4.res_block_0.downsample.conv')
load_bn(state_dict, ss_state_dict, 'layer4.0.downsample.1', 'stage_4.res_block_0.downsample.norm')

load_conv(state_dict, ss_state_dict, 'layer4.1.conv1', 'stage_4.res_block_1.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer4.1.bn1', 'stage_4.res_block_1.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer4.1.conv2', 'stage_4.res_block_1.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer4.1.bn2', 'stage_4.res_block_1.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer4.1.conv3', 'stage_4.res_block_1.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer4.1.bn3', 'stage_4.res_block_1.conv3.norm')

load_conv(state_dict, ss_state_dict, 'layer4.2.conv1', 'stage_4.res_block_2.conv1.conv')
load_bn(state_dict, ss_state_dict, 'layer4.2.bn1', 'stage_4.res_block_2.conv1.norm')
load_conv(state_dict, ss_state_dict, 'layer4.2.conv2', 'stage_4.res_block_2.conv2.conv')
load_bn(state_dict, ss_state_dict, 'layer4.2.bn2', 'stage_4.res_block_2.conv2.norm')
load_conv(state_dict, ss_state_dict, 'layer4.2.conv3', 'stage_4.res_block_2.conv3.conv')
load_bn(state_dict, ss_state_dict, 'layer4.2.bn3', 'stage_4.res_block_2.conv3.norm')

# save
torch.save(ss_state_dict, 'resnet50.pth')