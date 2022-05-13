import torch
import torchvision
import torch.nn as nn

class Hook():

    def __init__(self):
        self.module_name = []
        self.features_in_hook = []
        self.features_out_hook = []

    def __call__(self, module,fea_in,fea_out):
        #print("hooker working",self)
        self.module_name.append(module.__class__)
        self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)

    def clear(self):
        self.features_in_hook = []
        self.features_out_hook = []


class Attacker():
    def generate(self, wb_model, src, tgt):
        return None

    def __call__(self, wb_model, src, tgt):
        return self.generate(wb_model, src,tgt)

class AcivationAttacker(Attacker):
    def __init__(self, eps=0.07, k=10,library=None):#扰动大小，迭代次数
        self.eps = eps
        self.k = k
        self.handers = []
        self.library = library#目标图库

    def generate(self, wb_model, src, tgt):#白盒的模型，原图，目标图的标签

        wb_model.eval()
        adv = torch.Tensor(src.cpu()).to(src.device)
        adv.requires_grad = True
        alpha = self.eps / self.k
        momentum = torch.zeros(src.shape).to(src.device)
        #tgt1 = self.__get_furthest(wb_model, adv, target_label=tgt.item())
        with torch.no_grad():
            tgt_furthest = self.__get_furthest1(wb_model, adv, target_label=tgt.item())
            tgt_activation = self.__dense_hook(wb_model,tgt_furthest)

        for i in range(self.k):
            wb_model.zero_grad()
            loss = torch.dist(self.__dense_hook(wb_model,adv),
                              tgt_activation,
                              p=2)
            grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
            momentum = momentum + grad / torch.norm(grad, p=1)
            adv = torch.clip(adv - alpha * torch.sign(momentum), min=0, max=1)

        return adv

    def __dense_hook(self,densenet_121,img):#返回模型的中间层
        hook = Hook()
        #根据论文，这一层的效果最好
        hander = densenet_121.ormodel.features.denseblock4.denselayer14.register_forward_hook(hook)
        densenet_121.eval()
        _ = densenet_121(img)
        hander.remove()

        return hook.features_out_hook[0]

    def __get_furthest(self,model,ori_img,target_label):#返回库中最远的的目标图，采用切片的方式比较快
        lib = self.library
        images = torch.Tensor(lib[str(target_label)])
        images = torch.squeeze(images,1)
        q = self.__dense_hook(model,ori_img)
        p = self.__dense_hook(model,images)
        q = q.reshape(q.shape[0],-1)
        p = p.reshape(p.shape[0],-1)
        diff = q-p
        distance = torch.norm(diff,dim=1)
        best = torch.argmax(distance)

        return torch.unsqueeze(images[best],0)
