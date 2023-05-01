import pickle

class HALOConfig(object):
    def __init__(
            self,
            total_vocab_size=1619,
            code_vocab_size=1610,
            label_vocab_size=6,
            special_vocab_size=3,
            n_positions=45,
            n_ctx=45,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            semantic_w=0.01,
            batch_size=256,
            ccn_batch_size=32,
            sample_batch_size=1024,
            ccn_sample_batch_size=32,
            epoch=500,
            lr=1e-4,
            rules = pickle.load(open('inpatient_data/rules.pkl', 'rb'))
                    # [([], [], [], [1,6,8], [], 10, 1), # only current, positive
                    #  ([], [], [], [2,9,8], [4, 99], 12, 0), # only current, negative
                    #  ([], [], [], [3,6,8], [], 10, 1), # some nots in current
                    #  (-1, [2,7,20], [], [1,6,8], [], 11, 1), # all past
                    #  ([0, 1], [4,5,9], [], [1,6,8], [], 25, 1), # absolute past
                    #  ([-1, -2], [25, 76, 222], [], [1,6,8], [], 32, 1), # relative past
                    #  ([0, -1], [65, 77, 99], [], [1,6,8], [], 56, 1), # absolute and relative past
                    #  ([0], [32, 44], [], [], [], 10, 1)] # no current 
            
            # list of (which past visits, which positive codes from past visits, which negative codes from past visits, which positive codes in current visit, which negative codes in the current visit, which output code in current visit, value to set output code to)
            
    ):
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.label_vocab_size = label_vocab_size
        self.special_vocab_size = special_vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.semantic_w = semantic_w
        self.batch_size = batch_size
        self.ccn_batch_size = ccn_batch_size
        self.sample_batch_size = sample_batch_size
        self.ccn_sample_batch_size = ccn_sample_batch_size
        self.epoch = epoch
        self.lr = lr
        self.rules = rules