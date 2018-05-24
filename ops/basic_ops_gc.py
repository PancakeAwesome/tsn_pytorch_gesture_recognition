class ConsensusModule(torch.nn.Module):
    """docstring for ConsensusModule"""
    def __init__(self, consensus_type, dim = 1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim
        