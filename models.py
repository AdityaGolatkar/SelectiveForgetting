import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class NTK_Linear(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(NTK_Linear, self).__init__() 
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim,bias=False)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function

        out = self.linear(x)
        return out

class LinearNeuralTangentKernel(nn.Linear): 
    
    def __init__(self, in_features, out_features, bias=True, beta=np.sqrt(0.1), w_sig = np.sqrt(2.0)):
        self.beta = beta
        super(LinearNeuralTangentKernel, self).__init__(in_features, out_features)
        self.reset_parameters()
        self.w_sig = w_sig
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=1)

    def forward(self, input):
        return F.linear(input, self.w_sig * self.weight/np.sqrt(self.in_features), self.beta * self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, beta={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.beta)

class NTK_MLP(nn.Module):
    def __init__(self, num_classes=10, filters_percentage=1.0, beta=np.sqrt(0.1)):
        super(NTK_MLP, self).__init__()
        self.n_wid = int(32*filters_percentage)
        self.fc1 = LinearNeuralTangentKernel(1024, self.n_wid, beta=beta)
        self.fc2 = LinearNeuralTangentKernel(self.n_wid, num_classes, beta=beta)
#         self.fc3 = LinearNeuralTangentKernel(self.n_wid, self.n_wid, beta=beta)
#         self.fc4 = LinearNeuralTangentKernel(self.n_wid, self.n_wid, beta=beta)
#         self.fc5 = LinearNeuralTangentKernel(self.n_wid, num_classes, beta=beta)

    def forward(self, x):
        x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
        x = self.fc2(x)
        return x

class Affine(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        return x * self.weight + self.bias
    
class StandardLinearLayer(nn.Linear): 
    
    def __init__(self, in_features, out_features, bias=True, beta=np.sqrt(0.1), w_sig = np.sqrt(2.0)):
        self.beta = beta
        self.w_sig = w_sig
        super(StandardLinearLayer, self).__init__(in_features, out_features)
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=self.w_sig/np.sqrt(self.in_features))
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=self.beta)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, beta={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.beta)
    
class MLP(nn.Module):

    def __init__(self, num_layer=1, num_classes=10, filters_percentage=1., hidden_size=32, input_size=1024):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.num_layer = num_layer
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.layers = self._make_layers()

    def _make_layers(self):
        layer = []
        layer += [
            StandardLinearLayer(self.input_size,self.hidden_size),#nn.Linear(self.input_size, self.hidden_size),
            # Affine(self.hidden_size),
            nn.ReLU()]
        for i in range(self.num_layer - 2):
            layer += [StandardLinearLayer(self.hidden_size,self.hidden_size)]#[nn.Linear(self.hidden_size, self.hidden_size)]
            # layer += [Affine(self.hidden_size)]
            layer += [nn.ReLU()]
        layer += [StandardLinearLayer(self.hidden_size,self.num_classes)]#[nn.Linear(self.hidden_size, self.num_classes)]
        return nn.Sequential(*layer)

    def forward(self, x):
        x = x.reshape(x.size(0), self.input_size)
        return self.layers(x)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)
    
class ConvStandard(nn.Conv2d): 
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0, w_sig =\
                 np.sqrt(1.0)):
        super(ConvStandard, self).__init__(in_channels, out_channels,kernel_size)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.w_sig = w_sig
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=self.w_sig/(self.in_channels*np.prod(self.kernel_size)))
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)
            
    def forward(self, input):
        return F.conv2d(input,self.weight,self.bias,self.stride,self.padding)
            
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)

class AllCNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output
    
class ConvNeuralTangentKernel(nn.Conv2d): 
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0, w_sig =\
                 np.sqrt(1.0)):
        super(ConvNeuralTangentKernel, self).__init__(in_channels, out_channels,kernel_size)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.w_sig = w_sig
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)
            
    def forward(self, input):
        return F.conv2d(input, self.w_sig*self.weight/np.sqrt(self.in_channels*np.prod(self.kernel_size)),\
                        self.bias,self.stride,self.padding)
     
class ntk_Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
#         if not transpose:
        model += [ConvNeuralTangentKernel(in_channels,out_channels,kernel_size,stride=stride,padding=padding,
                                         output_padding=output_padding)] 
#         else:
#             model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
#                                          output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(ntk_Conv, self).__init__(*model)

class ntk_AllCNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(ntk_AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            ntk_Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            ntk_Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            ntk_Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            ntk_Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            ntk_Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            ntk_Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            ntk_Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            ntk_Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
#             LinearNeuralTangentKernel(n_filter2, num_classes, beta=np.sqrt(0.1)),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class _ResBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(_ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out
    
class ResNet18(nn.Module):
    def __init__(self, filters_percentage=1.0, n_channels = 3, num_classes=10, block=_ResBlock, num_blocks=[2,2,2,2], n_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(n_channels,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, int(64*filters_percentage), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*filters_percentage), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*filters_percentage), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*filters_percentage), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(512*filters_percentage)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class ResNet18_small(nn.Module):
    def __init__(self, filters_percentage=0.5, n_channels = 3, num_classes=10, block=_ResBlock, num_blocks=[2,2,2], n_classes=10):
        super(ResNet18_small, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(n_channels,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, int(64*filters_percentage), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*filters_percentage), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*filters_percentage), num_blocks[2], stride=2)
        self.linear = nn.Linear(int(256*filters_percentage)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out
    
class Wide_ResNet(nn.Module):
    def __init__(self, depth=4, filters_percentage=1, widen_factor=5, dropout_rate=0.0, num_classes=10):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

class ConvImprovedStandard(nn.Conv2d): 
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=(0,0), output_padding=0, w_sig =\
                 np.sqrt(2.0),s=10000):
        super(ConvImprovedStandard, self).__init__(in_channels, out_channels,kernel_size)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.w_sig = w_sig
        self.s = s
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1/np.sqrt(self.in_channels*np.prod(self.kernel_size)))
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)
            
    def forward(self, input):
        return F.conv2d(input, self.weight/np.sqrt(self.s),self.bias,self.stride,self.padding)

class wide_basicIS(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basicIS, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = ConvImprovedStandard(in_planes, planes, kernel_size=3, padding=(1,1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = ConvImprovedStandard(planes, planes, kernel_size=3, stride=stride, padding=(1,1))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                ConvImprovedStandard(in_planes, planes, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out
    
class Wide_ResNetIS(nn.Module):
    def __init__(self, depth=4, filters_percentage=1.0, widen_factor=1, dropout_rate=0.0, num_classes=10):
        super(Wide_ResNetIS, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = ConvImprovedStandard(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basicIS, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basicIS, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basicIS, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
    
class ConvNTK(nn.Conv2d): 
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=(0,0), output_padding=0, w_sig =\
                 np.sqrt(2.0)):
        super(ConvNTK, self).__init__(in_channels, out_channels,kernel_size)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.w_sig = w_sig
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)
            
    def forward(self, input):
        return F.conv2d(input, self.w_sig*self.weight/np.sqrt(self.in_channels*np.prod(self.kernel_size))\
                        ,self.bias,self.stride,self.padding)

class wide_basicNTK(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basicNTK, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = ConvNTK(in_planes, planes, kernel_size=3, padding=(1,1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = ConvNTK(planes, planes, kernel_size=3, stride=stride, padding=(1,1))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                ConvNTK(in_planes, planes, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out
    
class Wide_ResNetNTK(nn.Module):
    def __init__(self, depth=4, filters_percentage=1.0, widen_factor=1, dropout_rate=0.0, num_classes=10):
        super(Wide_ResNetNTK, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = ConvNTK(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basicNTK, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basicNTK, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basicNTK, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
    
_MODELS = {}

def _add_model(model_fn):
    _MODELS[model_fn.__name__] = model_fn
    return model_fn

@_add_model
def mlp(**kwargs):
    return MLP(**kwargs)

@_add_model
def ntk_linear(**kwargs):
    return NTK_Linear(**kwargs)

@_add_model
def ntk_mlp(**kwargs):
    return NTK_MLP(**kwargs)

@_add_model
def allcnn(**kwargs):
    return AllCNN(**kwargs)

@_add_model
def ntk_allcnn(**kwargs):
    return ntk_AllCNN(**kwargs)

@_add_model
def allcnn_no_bn(**kwargs):
    return AllCNN(batch_norm=False, **kwargs)

@_add_model
def resnet(**kwargs):
    return ResNet18(**kwargs)

@_add_model
def resnet_small(**kwargs):
    return ResNet18_small(**kwargs)

@_add_model
def wide_resnet(**kwargs):
    return Wide_ResNet(**kwargs)

@_add_model
def is_wide_resnet(**kwargs):
    return Wide_ResNetIS(**kwargs)

@_add_model
def ntk_wide_resnet(**kwargs):
    return Wide_ResNetNTK(**kwargs)

def get_model(name, **kwargs):
    return _MODELS[name](**kwargs)
