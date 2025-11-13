import torch as th

# class FuzeParameter(th.nn.Module)::
#     def __init__(self, pretrained_param: th.nn.Parameter, new_param: th.nn.Parameter, alpha=0.5):
#         super(FuzeParameter, self).__init__()
#         self.pretrained_param = pretrained_param
#         self.new_param = new_param
#         self.alpha = alpha
    
#     def forward(self):
#         return self.alpha * self.pretrained_param + (1 - self.alpha) * self.new_param

class FuzeLayer(th.nn.Module):
    def __init__(self, pretrained_layer: th.nn.Module, new_layer: th.nn.Module, alpha=1.0):
        super(FuzeLayer, self).__init__()
        self.pretrained_layer = pretrained_layer
        self.new_layer = new_layer
        self.alpha = alpha
        # self.alpha = th.nn.Parameter(th.Tensor([alpha]), requires_grad=True)
        self._sanity_check()
        self.pretrained_layer.requires_grad_(False)
    
    def _sanity_check(self):
        if self.pretrained_layer.__class__ != self.new_layer.__class__:
            raise ValueError(f"The two layers must be of the same type, \
                             get {self.pretrained_layer.__class__} and {self.new_layer.__class__}.")
        # if self.pretrained_layer.device != self.new_layer.device:
        #     raise ValueError("The two layers must be on the same device.")
        pretrained_params = {n: p.shape for n, p in self.pretrained_layer.named_parameters()}
        new_params = {n: p.shape for n, p in self.new_layer.named_parameters()}
        if pretrained_params.keys() != new_params.keys():
            raise ValueError("The two layers must have the same parameters.")
        for p1, p2 in zip(pretrained_params.values(), new_params.values()):
            if p1 != p2:
                raise ValueError(f"The two layers must have the same parameter shapes, get {p1} and {p2}.")


    def forward(self, *args, **kwargs):
        '''
        Premitive version: simply add the outputs of two layers.
        TODO: implement a more advanced fusion strategy.
        '''
        pretrained_output = self.pretrained_layer(*args, **kwargs)
        new_output = self.new_layer(*args, **kwargs)
        return pretrained_output + self.alpha * new_output
