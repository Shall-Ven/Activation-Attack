import torchvision.transforms as T
import torchvision
import torch

from Models import densenet
from Models import resnet
from Models import vgg
from Tool import predict_image, save_variable, wr_json
import torch.nn as nn
from collections import OrderedDict

class Normalize(nn.Module):
    def __init__(self,dataset_name):
        super(Normalize, self).__init__()
        assert dataset_name in ['imagenet', 'cifar10',None], 'check dataset_name'
        if dataset_name == 'imagenet':
            self.normalize = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
        elif dataset_name == 'cifar10':
            self.normalize = [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)]
        elif dataset_name == None:
            self.normalize = [(0, 0, 0), (1, 1, 1)]
    def forward(self, input):
        x = input.clone()
        for i in range(x.shape[1]):
            x[:,i] = (x[:,i] - self.normalize[0][i]) / self.normalize[1][i]
        return x


map_location = None
densenet121 = densenet.densenet121()
resnet50 = resnet.resnet50()
vgg19bn = vgg.vgg19_bn()
densenet121.eval()
resnet50.eval()
vgg19bn.eval()

state_dicts_densenet121 = torch.load("./Models/state_dicts/densenet121.pt",map_location=map_location)
densenet121.load_state_dict(state_dicts_densenet121,False)
densenet121.eval()

state_dicts_resnet50 = torch.load("./Models/state_dicts/resnet50.pt",map_location=map_location)
resnet50.load_state_dict(state_dicts_resnet50)
resnet50.eval()

state_dicts_vggg19bn = torch.load("./Models/state_dicts/vgg19_bn.pt",map_location=map_location)
vgg19bn.load_state_dict(state_dicts_vggg19bn)
vgg19bn.eval()


transform = T.Compose([T.ToTensor(),])#T.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2471, 0.2435, 0.2616])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)

normal = 'cifar10'
densenet_121 = nn.Sequential(OrderedDict([('normal',Normalize(normal)), ('ormodel',densenet121.eval())]))
vgg19_bn = nn.Sequential(OrderedDict([('normal',Normalize(normal)), ('ormodel',vgg19bn.eval())]))
resnet_50 = nn.Sequential(OrderedDict([('normal',Normalize(normal)), ('ormodel',resnet50.eval())]))
densenet_121.eval()
vgg19_bn.eval()
resnet_50.eval()

def get_lib():
    data = dict(zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [[]] * 10))

    for i,(images,label) in enumerate(testloader):

        flag = 0

        for i in range(10):
            if len(data[i]) == 100:
                flag += 1
        if flag == 10:
            break

        if predict_image(densenet_121,images)!=label.item():
            continue
        if predict_image(resnet_50,images)!=label.item():
            continue
        if predict_image(vgg19_bn,images)!=label.item():
            continue
        if len(data[label.item()]) == 100:
            continue

        temp = data[label.item()].copy()
        temp.append(images.detach().numpy().tolist())
        data.update({label.item():temp})

        wr_json(data={label.item(): data[label.item()]},  # 代保存字典文件
                path_file='./library/library.json',
                type='修改添加'  # 读写方式
                )

    return data

if __name__ == "__main__":
    get_lib()




