class Fusion_Branch(nn.Module):
    def __init__(self, input_size, output_size):
        super(Fusion_Branch, self).__init__()

        self.fc = nn.Linear(input_size, output_size)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, global_pool, local_pool):
        #fusion = torch.cat((global_pool.unsqueeze(2), local_pool.unsqueeze(2)), 2).cuda()
        #fusion = fusion.max(2)[0]#.squeeze(2).cuda()
        #print(fusion.shape)
        fusion = torch.cat((global_pool,local_pool), 1).cuda()
        fusion_var = torch.autograd.Variable(fusion)
        x = self.fc(fusion_var)
        x = self.Sigmoid(x)
        return x