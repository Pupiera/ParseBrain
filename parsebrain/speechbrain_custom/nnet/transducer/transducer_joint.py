import torch
import torch.nn as nn

class TransducerJointMultitask(nn.Module):
    def __init__(self, joint_network=None, joint="sum", nonlinearity=torch.nn.LeakyReLU):
        super().__init__()
        self.joint = joint
        self.joint_network = joint_network
        self.nonlinearity = nonlinearity

    def forward(self, list_input: list[torch.Tensor]):
        if self.joint == "concat":
            raise NotImplementedError()
        if self.joint =="sum":
            output = list_input[0]
            assert type(output) == torch.Tensor
            for x in list_input[1:]:
                output+= x
        return output


    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        self.joint_network(first_input)
