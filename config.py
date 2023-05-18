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
            batch_size=32,
            sample_batch_size=32,
            epoch=500,
            lr=1e-4,
            rules = pickle.load(open('./inpatient_data/rules.pkl', 'rb'))            
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
        self.sample_batch_size = sample_batch_size
        self.epoch = epoch
        self.lr = lr
        self.rules = rules