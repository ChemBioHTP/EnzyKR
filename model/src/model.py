import torch
import torch.nn as nn
import torch.nn.functional as F
from bidirectional_cross_attention import BidirectionalCrossAttention
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict



#The residual block for the encoder

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class DistanceRepresentation(nn.Module):
    def __init__(self, input_channels=1, input_length=1254, num_classes=96):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.relu = nn.ReLU()
        #self.fc = nn.Linear(128 * input_length, num_classes)
        self.fc = nn.Linear(128 * input_length, 96) # to make output input the [64,33 ]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.unsqueeze(-1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MSAencoder(nn.Module):
    def __init__(self, input_channels=26, output_channels=32):
        super(MSAencoder, self).__init__()
        self.embed = nn.Embedding(128, 128, padding_idx=0)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(128 * 1254 * 10, 96)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 3, 1, 2)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear(out)
        return out

class DistEncoder(nn.Module):
    def __init__(self):
        super(DistEncoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=10) # (64, 1, 125, 125)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2, 3)
        x = self.conv(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x

class NodeLevelBatchNorm(_BatchNorm):
    r"""
    Applies Batch Normalization over a batch of graph data.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    batch_nodes_dim: all nodes of a batch graph
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)

class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels).jittable()
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = F.relu(self.norm(self.conv(x, edge_index)))

        return data

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features

        data = self.conv1(data)

        return data

    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]

        data = self.bn_function(data)
        data = self.conv2(data)

        return data

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)


    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features

        data.x = torch.cat(data.x, 1)

        return data


class GraphDenseNet(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config = (3, 3, 3, 3), bn_sizes=[2, 3, 4, 4]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i+1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, num_input_features // 2)
            self.features.add_module("transition%d" % (i+1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        data = self.features(data)
        x = gnn.global_mean_pool(data.x, data.batch)
        x = self.classifer(x)

        return x

class KcatRegressor(nn.Module):
    def __init__(self, inputs_dim, context_dims):
        super(KcatRegressor, self).__init__()
        self.cross_attn = BidirectionalCrossAttention(dim = inputs_dim,
                                                      heads=8, dim_head=8,
                                                      context_dim=context_dims).to(torch.float64)
        self.residual1 = ResidualBlock(1, 1).to(torch.float64)
        self.fc1 = nn.Linear((inputs_dim+context_dims), 256).to(torch.float64)
        self.fc2 = nn.Linear(256, 1).to(torch.float64)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x2 = x2.view(x1.size(0), 1, -1)
        x1 = x1.to(torch.float64)
        x1, x2 = self.cross_attn(x1, x2, mask=None, context_mask=None)
        x = torch.cat((x1, x2), dim=2)
        out = self.relu(self.residual1(x))
        out = out.view(out.size(0), -1)
        out = out.to(torch.float64)
        out = self.relu(self.fc1(out))
        out =  self.fc2(out)
        return out

class EnzyKR(nn.Module):
    def __init__(self, embedding_size=128, filter_num=32, out_dim=1):
        super().__init__()
        self.MSAencoder = MSAencoder(1, 32)
        self.avg_dist_encoder = DistanceRepresentation()
        self.ligand_encoder = GraphDenseNet(num_input_features=22, out_dim=filter_num*3, block_config=[8, 8, 8], bn_sizes=[2, 2, 2])
        self.distencoder = DistEncoder().to(torch.float64)
        self.regressor = KcatRegressor(inputs_dim=(filter_num + 16) * 3 * 2,
                                       context_dims = 125*125)
        self.classifier = nn.Sequential(
            nn.Linear((filter_num + 16) * 3 * 2, 1024), #add 16 to fit the avg dist
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, data):
        msas = data.msas
        dists = data.distance_map
        avg_dists = data.avg_distance_map
        protein_x = self.MSAencoder(msas) #speedup
        avg_dist_x = self.avg_dist_encoder(avg_dists)
        ligand_x = self.ligand_encoder(data)
        x = torch.cat([protein_x, avg_dist_x, ligand_x], dim=-1)
        dists_x = self.distencoder(dists)
        out = self.regressor(x, dists_x)
        #x = self.classifier(x)
        return out

        for layer in self.regressor.cross_attn.modules():
            if isinstance(layer, nn.Linear, Conv1d):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        for layer in self.MSAencoder.modules():
            if isinstance(layer, nn.Conv2d):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

            elif isinstance(layer, nn.Conv1d):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)


