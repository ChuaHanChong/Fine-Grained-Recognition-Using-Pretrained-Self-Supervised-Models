import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Union
import copy


class GCNCombiner(nn.Module):

    def __init__(
        self,
        total_num_selects: int,
        num_classes: int,
        inputs: Union[dict, None] = None,
        proj_size: Union[int, None] = None,
        fpn_size: Union[int, None] = None,
    ):
        """
        If building backbone without FPN, set fpn_size to None and MUST give
        'inputs' and 'proj_size', the reason of these setting is to constrain the
        dimension of graph convolutional network input.
        """
        super(GCNCombiner, self).__init__()

        assert (
            inputs is not None or fpn_size is not None
        ), "To build GCN combiner, you must give one features dimension."

        ### auto-proj
        self.fpn_size = fpn_size
        if fpn_size is None:
            for name in inputs:
                if len(name) == 4:
                    in_size = inputs[name].size(1)
                elif len(name) == 3:
                    in_size = inputs[name].size(2)
                else:
                    raise ValueError("The size of output dimension of previous must be 3 or 4.")

                m = nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size),
                )
                self.add_module("proj_" + name, m)
            self.proj_size = proj_size
        else:
            self.proj_size = fpn_size

        ### build one layer structure (with adaptive module)
        num_joints = total_num_selects // 64

        self.param_pool0 = nn.Linear(total_num_selects, num_joints)  # [B, fpn_size, total_num_selects] -> [B, fpn_size, num_joints]

        A = torch.eye(num_joints) / 100 + 1 / 100
        self.adj1 = nn.Parameter(copy.deepcopy(A))  # [num_joints, num_joints]
        self.conv1 = nn.Conv1d(self.proj_size, self.proj_size, 1)  # [B, proj_size, num_joints] -> [B, proj_size, num_joints]
        self.batch_norm1 = nn.BatchNorm1d(self.proj_size)  # [proj_size]

        self.conv_q1 = nn.Conv1d(self.proj_size, self.proj_size // 4, 1)  # [B, proj_size, num_joints] -> [B, proj_size // 4, num_joints]
        self.conv_k1 = nn.Conv1d(self.proj_size, self.proj_size // 4, 1)  # [B, proj_size, num_joints] -> [B, proj_size // 4, num_joints]
        self.alpha1 = nn.Parameter(torch.zeros(1))

        ### merge information
        self.param_pool1 = nn.Linear(num_joints, 1)  # [B, fpn_size, num_joints] -> [B, fpn_size, 1]

        #### class predict
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.proj_size, num_classes)  # [B, fpn_size, 1] -> [B, num_classes]

        self.tanh = nn.Tanh()

    def forward(self, x):
        """ """
        hs = []  # [[B, num_selected, fpn_size], ...]
        names = []
        for name in x:
            if "FPN1_" in name:
                continue
            if self.fpn_size is None:
                _tmp = getattr(self, "proj_" + name)(x[name])
            else:
                _tmp = x[name]
            hs.append(_tmp)
            names.append([name, _tmp.size()])

        hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous()  # hs = [[B, num_selected, fpn_size], ...] -> [B, total_num_selects, fpn_size] -> [B, fpn_size, total_num_selects)]
        # print(hs.size(), names)
        hs = self.param_pool0(hs)  # hs = [B, fpn_size, total_num_selects] -> [B, fpn_size, num_joints]
        ### adaptive adjacency
        q1 = self.conv_q1(hs).mean(1)  # hs = [B, fpn_size, num_joints] -> [B, fpn_size // 4, num_joints] -> [B, num_joints]
        k1 = self.conv_k1(hs).mean(1)  # k1 = [B, fpn_size, num_joints] -> [B, fpn_size // 4, num_joints] -> [B, num_joints]
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))  # q1 = [B, num_joints] -> [B, num_joints, 1], k1 = [B, num_joints] -> [B, 1, num_joints], A1 = [B, num_joints, num_joints], q1 - k1 (computes pairwise differences)
        A1 = self.adj1 + A1 * self.alpha1  # A1 = [num_joints, num_joints] + [B, num_joints, num_joints] * [1]
        ### graph convolution
        hs = self.conv1(hs)  # hs = [B, fpn_size, num_joints] -> [B, fpn_size, num_joints]
        hs = torch.matmul(hs, A1)  # hs = [B, fpn_size, num_joints] @ [B, num_joints, num_joints] -> [B, fpn_size, num_joints]
        hs = self.batch_norm1(hs)  # hs = [B, fpn_size, num_joints]
        ### predict
        hs = self.param_pool1(hs)  # hs = [B, fpn_size, num_joints] -> [B, fpn_size, 1]
        hs = self.dropout(hs)
        hs = hs.flatten(1)  # hs = [B, fpn_size, 1] -> [B, fpn_size]
        hs = self.classifier(hs)  # hs = [B, fpn_size] -> [B, num_classes]

        return hs

# The adaptive adjacency mechanism in the GCNCombiner class dynamically adjusts the adjacency matrix used in the graph convolution operation based on the input features. 
# This allows the model to learn and adapt the relationships between different nodes in the graph during training. 

# Purpose of Adaptive Adjacency
# The adaptive adjacency mechanism allows the model to learn the relationships between nodes dynamically based on the input features. 
# This is particularly useful in graph convolutional networks (GCNs) where the structure of the graph (i.e., the connections between nodes) can significantly impact the performance of the model. 
# By learning an adaptive adjacency matrix, the model can better capture the underlying structure of the data and improve its ability to make accurate predictions.

# Summary
# The adaptive adjacency mechanism in the GCNCombiner class involves computing a dynamic adjacency matrix based on the input features using query and key convolutions. 
# This matrix is then used in the graph convolution operation to propagate information between nodes, allowing the model to learn and adapt the relationships between nodes during training.


class WeaklySelector(nn.Module):

    def __init__(
        self,
        inputs: dict,
        num_classes: int,
        num_select: dict,
        fpn_size: Union[int, None] = None,
    ):
        """
        inputs: dictionary contain torch.Tensors, which comes from backbone
                [Tensor1(hidden feature1), Tensor2(hidden feature2)...]
                Please note that if len(features.size) equal to 3, the order of dimension must be [B,S,C],
                S mean the spatial domain, and if len(features.size) equal to 4, the order must be [B,C,H,W]
        """
        super(WeaklySelector, self).__init__()

        self.num_select = num_select

        self.fpn_size = fpn_size
        ### build classifier
        if self.fpn_size is None:
            self.num_classes = num_classes
            for name in inputs:
                fs_size = inputs[name].size()
                if len(fs_size) == 3:
                    in_size = fs_size[2]
                elif len(fs_size) == 4:
                    in_size = fs_size[1]
                m = nn.Linear(in_size, num_classes)
                self.add_module("classifier_l_" + name, m)

        self.thresholds = {}
        for name in inputs:
            self.thresholds[name] = []

    # def select(self, logits, l_name):
    #     """
    #     logits: [B, S, num_classes]
    #     """
    #     probs = torch.softmax(logits, dim=-1)
    #     scores, _ = torch.max(probs, dim=-1)
    #     _, ids = torch.sort(scores, -1, descending=True)
    #     sn = self.num_select[l_name]
    #     s_ids = ids[:, :sn]
    #     not_s_ids = ids[:, sn:]
    #     return s_ids.unsqueeze(-1), not_s_ids.unsqueeze(-1)

    def forward(self, x, logits=None):
        """
        x :
            dictionary contain the features maps which
            come from your choosen layers.
            size must be [B, HxW, C] ([B, S, C]) or [B, C, H, W].
            [B,C,H,W] will be transpose to [B, HxW, C] automatically.
        """
        if self.fpn_size is None:
            logits = {}
        selections = {}
        for name in x:
            # print("[selector]", name, x[name].size())
            if "FPN1_" in name:
                continue
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                x[name] = x[name].view(B, C, H * W).permute(0, 2, 1).contiguous()
            C = x[name].size(-1)
            if self.fpn_size is None:
                logits[name] = getattr(self, "classifier_l_" + name)(x[name])

            probs = torch.softmax(logits[name], dim=-1)  # probs = [B, N, num_classes]
            sum_probs = torch.softmax(logits[name].mean(1), dim=-1)  # sum_probs = [B, num_classes]
            selections[name] = []
            preds_1 = []
            preds_0 = []
            num_select = self.num_select[name]
            for bi in range(logits[name].size(0)):
                _, max_ids = torch.max(sum_probs[bi], dim=-1)
                confs, ranks = torch.sort(probs[bi, :, max_ids], descending=True)  # confs = [N], ranks = [N]
                sf = x[name][bi][ranks[:num_select]]  # sf = [32, C]
                nf = x[name][bi][ranks[num_select:]]  # calculate  # nf = [N-32, C]
                selections[name].append(sf)  # [num_selected, C] -> [[num_selected, C], ...]
                preds_1.append(logits[name][bi][ranks[:num_select]])  # preds_1 = [num_selected, num_classes]
                preds_0.append(logits[name][bi][ranks[num_select:]])  # preds_0 = [N-32, num_classes]

                if bi >= len(self.thresholds[name]):
                    self.thresholds[name].append(confs[num_select])  # for initialize  # e.g. {'layer1': [tensor(0.1054), ...], 'layer2': [tensor(0.1085), ...], 'layer3': [...], 'layer4': [...]}
                else:
                    self.thresholds[name][bi] = confs[num_select]

            selections[name] = torch.stack(selections[name])  # [B, num_selected, C]
            preds_1 = torch.stack(preds_1)  # preds_1 = [B, num_selected, num_classes]
            preds_0 = torch.stack(preds_0)  # preds_0 = [B, N-32, num_classes]

            logits["select_" + name] = preds_1
            logits["drop_" + name] = preds_0

        return selections


class FPN(nn.Module):

    def __init__(self, inputs: dict, fpn_size: int, proj_type: str, upsample_type: str):
        """
        inputs : dictionary contains torch.Tensor
                 which comes from backbone output
        fpn_size: integer, fpn
        proj_type:
            in ["Conv", "Linear"]
        upsample_type:
            in ["Bilinear", "Conv", "Fc"]
            for convolution neural network (e.g. ResNet, EfficientNet), recommand 'Bilinear'.
            for Vit, "Fc". and Swin-T, "Conv"
        """
        super(FPN, self).__init__()
        assert proj_type in ["Conv", "Linear"], "FPN projection type {} were not support yet, please choose type 'Conv' or 'Linear'".format(proj_type)
        assert upsample_type in ["Bilinear", "Conv"], "FPN upsample type {} were not support yet, please choose type 'Bilinear' or 'Conv'".format(upsample_type)

        self.fpn_size = fpn_size
        self.upsample_type = upsample_type
        inp_names = [name for name in inputs]

        for i, node_name in enumerate(inputs):
            ### projection module
            if proj_type == "Conv":
                m = nn.Sequential(
                    nn.Conv2d(inputs[node_name].size(1), inputs[node_name].size(1), 1),  # [B, C, H, W] -> [B, C, H, W]
                    nn.ReLU(),
                    nn.Conv2d(inputs[node_name].size(1), fpn_size, 1),  # [B, C, H, W] -> [B, fpn_size, H, W]
                )
            elif proj_type == "Linear":
                m = nn.Sequential(
                    nn.Linear(inputs[node_name].size(-1), inputs[node_name].size(-1)),  # [B, N, C] -> [B, N, C]
                    nn.ReLU(),
                    nn.Linear(inputs[node_name].size(-1), fpn_size),  # [B, N, C] -> [B, N, fpn_size]
                )
            self.add_module("Proj_" + node_name, m)

            ### upsample module
            if upsample_type == "Conv" and i != 0:
                assert len(inputs[node_name].size()) == 3  # B, S, C
                in_dim = inputs[node_name].size(1)
                out_dim = inputs[inp_names[i - 1]].size(1)
                # if in_dim != out_dim:
                m = nn.Conv1d(in_dim, out_dim, 1)  # for spatial domain  # [B, N_in, C] -> [B, N_out, C]
                # else:
                #     m = nn.Identity()
                self.add_module("Up_" + node_name, m)

        if upsample_type == "Bilinear":
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def upsample_add(self, x0: torch.Tensor, x1: torch.Tensor, x1_name: str):
        """
        return Upsample(x1) + x1
        """
        if self.upsample_type == "Bilinear":
            if x1.size(-1) != x0.size(-1):
                x1 = self.upsample(x1)
        else:
            x1 = getattr(self, "Up_" + x1_name)(x1)  #  x1 = [B, N_in, C] -> [B, N_out, C]
        return x1 + x0

    def forward(self, x):
        """
        x : dictionary
            {
                "node_name1": feature1,
                "node_name2": feature2, ...
            }
        """
        ### project to same dimension
        hs = []  # hs = ['layer1', ..., 'layer4']
        for i, name in enumerate(x):
            if "FPN1_" in name:
                continue
            x[name] = getattr(self, "Proj_" + name)(x[name])  # [B, N, C] -> [B, N, fpn_size] (linear projection)
            hs.append(name)

        x["FPN1_" + "layer4"] = x["layer4"]  # x = {'layer1': [B, N, fpn_size], ..., 'FPN1_layer4': [B, N, fpn_size]}

        for i in range(len(hs) - 1, 0, -1):
            x1_name = hs[i]
            x0_name = hs[i - 1]
            x[x0_name] = self.upsample_add(x[x0_name], x[x1_name], x1_name)  # x1 = [B, N_in, fpn_size] -> [B, N_out, fpn_size]
            x["FPN1_" + x0_name] = x[x0_name]  # x = {'FPN1_layer4': [B, N_out, fpn_size], 'FPN1_layer3': [...], ...}

        return x


class FPN_UP(nn.Module):

    def __init__(self, inputs: dict, fpn_size: int):
        super(FPN_UP, self).__init__()

        inp_names = [name for name in inputs]

        for i, node_name in enumerate(inputs):
            ### projection module
            m = nn.Sequential(
                nn.Linear(fpn_size, fpn_size),  # [B, N, fpn_size] -> [B, N, fpn_size]
                nn.ReLU(),
                nn.Linear(fpn_size, fpn_size),  # [B, N, fpn_size] -> [B, N, fpn_size]
            )  # [B, N, fpn_size] -> [B, N, fpn_size]
            self.add_module("Proj_" + node_name, m)

            ### upsample module
            if i != (len(inputs) - 1):
                assert len(inputs[node_name].size()) == 3  # B, S, C
                in_dim = inputs[node_name].size(1)
                out_dim = inputs[inp_names[i + 1]].size(1)
                m = nn.Conv1d(in_dim, out_dim, 1)  # for spatial domain  # [B, N_in, C] -> [B, N_out, C]
                self.add_module("Down_" + node_name, m)
                # print("Down_"+node_name, in_dim, out_dim)
                """
                Down_layer1 2304 576
                Down_layer2 576 144
                Down_layer3 144 144
                """

    def downsample_add(self, x0: torch.Tensor, x1: torch.Tensor, x0_name: str):
        """
        return Upsample(x1) + x1
        """
        # print("[downsample_add] Down_" + x0_name)
        x0 = getattr(self, "Down_" + x0_name)(x0)  # x0 = [B, N_in, fpn_size] -> [B, N_out, fpn_size]
        return x1 + x0

    def forward(self, x):
        """
        x : dictionary
            {
                "node_name1": feature1,
                "node_name2": feature2, ...
            }
        """
        ### project to same dimension
        hs = []  # hs = ['layer1', ..., 'layer4']
        for i, name in enumerate(x):
            if "FPN1_" in name:
                continue
            x[name] = getattr(self, "Proj_" + name)(x[name])  # [B, N, fpn_size] -> [B, N, fpn_size]  (linear projection)
            hs.append(name)

        # print(hs)
        for i in range(0, len(hs) - 1):
            x0_name = hs[i]
            x1_name = hs[i + 1]
            # print(x0_name, x1_name)
            # print(x[x0_name].size(), x[x1_name].size())
            x[x1_name] = self.downsample_add(x[x0_name], x[x1_name], x0_name)  # x = {'layer1': [B, N_out, fpn_size], ...}
        return x


class PluginMoodel(nn.Module):

    def __init__(
        self,
        backbone: torch.nn.Module,
        return_nodes: Union[dict, None],
        img_size: int,
        use_fpn: bool,
        fpn_size: Union[int, None],
        proj_type: str,
        upsample_type: str,
        use_selection: bool,
        num_classes: int,
        num_selects: dict,
        use_combiner: bool,
        comb_proj_size: Union[int, None],
    ):
        """
        * backbone:
            torch.nn.Module class (recommand pretrained on ImageNet or IG-3.5B-17k(provided by FAIR))
        * return_nodes:
            e.g.
            return_nodes = {
                # node_name: user-specified key for output dict
                'layer1.2.relu_2': 'layer1',
                'layer2.3.relu_2': 'layer2',
                'layer3.5.relu_2': 'layer3',
                'layer4.2.relu_2': 'layer4',
            } # you can see the example on https://pytorch.org/vision/main/feature_extraction.html
            !!! if using 'Swin-Transformer', please set return_nodes to None
            !!! and please set use_fpn to True
        * feat_sizes:
            tuple or list contain features map size of each layers.
            ((C, H, W)). e.g. ((1024, 14, 14), (2048, 7, 7))
        * use_fpn:
            boolean, use features pyramid network or not
        * fpn_size:
            integer, features pyramid network projection dimension
        * num_selects:
            num_selects = {
                # match user-specified in return_nodes
                "layer1": 2048,
                "layer2": 512,
                "layer3": 128,
                "layer4": 32,
            }
        Note: after selector module (WeaklySelector) , the feature map's size is [B, S', C] which
        contained by 'logits' or 'selections' dictionary (S' is selection number, different layer
        could be different).
        """
        super(PluginMoodel, self).__init__()

        ### = = = = = Backbone = = = = =
        self.return_nodes = return_nodes
        if return_nodes is not None:
            self.backbone = create_feature_extractor(
                backbone, return_nodes=return_nodes
            )
        else:
            self.backbone = backbone

        ### get hidden feartues size
        rand_in = torch.randn(1, 3, img_size, img_size)
        outs = self.backbone(rand_in)

        ### just original backbone
        if not use_fpn and (not use_selection and not use_combiner):
            for name in outs:
                fs_size = outs[name].size()
                if len(fs_size) == 3:
                    out_size = fs_size.size(-1)
                elif len(fs_size) == 4:
                    out_size = fs_size.size(1)
                else:
                    raise ValueError("The size of output dimension of previous must be 3 or 4.")
            self.classifier = nn.Linear(out_size, num_classes)

        ### = = = = = FPN = = = = =
        self.use_fpn = use_fpn
        if self.use_fpn:
            self.fpn_down = FPN(outs, fpn_size, proj_type, upsample_type)
            self.build_fpn_classifier_down(outs, fpn_size, num_classes)
            self.fpn_up = FPN_UP(outs, fpn_size)
            self.build_fpn_classifier_up(outs, fpn_size, num_classes)

        self.fpn_size = fpn_size

        ### = = = = = Selector = = = = =
        self.use_selection = use_selection
        if self.use_selection:
            w_fpn_size = (
                self.fpn_size if self.use_fpn else None
            )  # if not using fpn, build classifier in weakly selector
            self.selector = WeaklySelector(outs, num_classes, num_selects, w_fpn_size)

        ### = = = = = Combiner = = = = =
        self.use_combiner = use_combiner
        if self.use_combiner:
            assert self.use_selection, "Please use selection module before combiner"
            if self.use_fpn:
                gcn_inputs, gcn_proj_size = None, None
            else:
                gcn_inputs, gcn_proj_size = (
                    outs,
                    comb_proj_size,
                )  # redundant, fix in future
            total_num_selects = sum([num_selects[name] for name in num_selects])  # sum
            self.combiner = GCNCombiner(
                total_num_selects, num_classes, gcn_inputs, gcn_proj_size, self.fpn_size
            )

    def build_fpn_classifier_up(self, inputs: dict, fpn_size: int, num_classes: int):
        """
        Teh results of our experiments show that linear classifier in this case may cause some problem.
        """
        for name in inputs:
            m = nn.Sequential(
                nn.Conv1d(fpn_size, fpn_size, 1),  # [B, fpn_size, N] -> [B, fpn_size, N]
                nn.BatchNorm1d(fpn_size),
                nn.ReLU(),
                nn.Conv1d(fpn_size, num_classes, 1),  # [B, fpn_size, N] -> [B, num_classes, N]
            )  # [B, fpn_size, N] -> [B, num_classes, N]
            self.add_module("fpn_classifier_up_" + name, m)

    def build_fpn_classifier_down(self, inputs: dict, fpn_size: int, num_classes: int):
        """
        Teh results of our experiments show that linear classifier in this case may cause some problem.
        """
        for name in inputs:
            m = nn.Sequential(
                nn.Conv1d(fpn_size, fpn_size, 1),  # [B, fpn_size, N] -> [B, fpn_size, N]
                nn.BatchNorm1d(fpn_size),
                nn.ReLU(),
                nn.Conv1d(fpn_size, num_classes, 1),  # [B, fpn_size, N] -> [B, num_classes, N]
            )  # [B, fpn_size, N] -> [B, num_classes, N]
            self.add_module("fpn_classifier_down_" + name, m)

    def forward_backbone(self, x):
        return self.backbone(x)

    def fpn_predict_down(self, x: dict, logits: dict):
        """
        x: [B, C, H, W] or [B, S, C]
           [B, C, H, W] --> [B, H*W, C]
        """
        for name in x:
            if "FPN1_" not in name:
                continue
            ### predict on each features point
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H * W)
            elif len(x[name].size()) == 3:
                logit = x[name].transpose(1, 2).contiguous()  # logit = [B, N, C] -> [B, C, N]
            model_name = name.replace("FPN1_", "")
            logits[name] = getattr(self, "fpn_classifier_down_" + model_name)(logit)  # logits = {'FPN1_layer4': [B, num_classes, N], ...}
            logits[name] = logits[name].transpose(1, 2).contiguous()  # transpose  # logits = {'FPN1_layer4': [B, num_classes, N] -> [B, N, num_classes], ...}

    def fpn_predict_up(self, x: dict, logits: dict):
        """
        x: [B, C, H, W] or [B, S, C]
           [B, C, H, W] --> [B, H*W, C]
        """
        for name in x:
            if "FPN1_" in name:
                continue
            ### predict on each features point
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H * W)
            elif len(x[name].size()) == 3:
                logit = x[name].transpose(1, 2).contiguous()  # logit = [B, N, C] -> [B, C, N]
            model_name = name.replace("FPN1_", "")
            logits[name] = getattr(self, "fpn_classifier_up_" + model_name)(logit)  # logits = {'layer1': [B, num_classes, N], ...}
            logits[name] = logits[name].transpose(1, 2).contiguous()  # transpose  # logits = {'layer1': [B, num_classes, N] -> [B, N, num_classes], ...}

    def forward(self, x: torch.Tensor):

        logits = {}

        x = self.forward_backbone(x)  # x = [B, 3, H, W] -> {'layer1': [B, N, C], ...} (ViT)

        if self.use_fpn:
            x = self.fpn_down(x)  # x = {'layer1': [B, N, fpn_size], ..., 'FPN1_layer4': [B, N, fpn_size], ...}, layer1 == FPN1_layer1
            # print([name for name in x])
            self.fpn_predict_down(x, logits)  # logits = {'FPN1_layer4': [B, N, num_classes], ...}
            x = self.fpn_up(x)  # x = {'layer1': [B, N, fpn_size], ..., 'FPN1_layer4': [B, N, fpn_size], ...}
            self.fpn_predict_up(x, logits)  # logits = {'FPN1_layer4': [B, N, num_classes], ..., 'layer1': [B, N, num_classes], ...}

        if self.use_selection:
            selects = self.selector(x, logits)

        if self.use_combiner:
            comb_outs = self.combiner(selects)
            logits["comb_outs"] = comb_outs
            return logits

        if self.use_selection or self.use_fpn:
            return logits

        ### original backbone (only predict final selected layer)
        for name in x:
            hs = x[name]

        if len(hs.size()) == 4:
            hs = F.adaptive_avg_pool2d(hs, (1, 1))
            hs = hs.flatten(1)
        else:
            hs = hs.mean(1)
        out = self.classifier(hs)
        logits["ori_out"] = out

        return logits
