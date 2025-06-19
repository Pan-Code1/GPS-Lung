class Config:
    def __init__(self):
        # data
        self.minHU = -1000
        self.maxHU = 500
        self.temp_path = "data/temp" # "/path/to/temp"
        self.input_size = (512, 512)

        # communication
        self.server_url = "/CBCT2CT"
        self.host = "127.0.0.1"
        self.port = 6789
        
        # model
        self.model_path = "pretrained_model" # path/to/weight
        self.device = "cuda"

class Arguments:
    def __init__(self):
        self.seed = 1024
        self.compute_fid = False
        self.epoch_id = 1000
        self.num_channels = 2
        self.centered = True
        self.use_geometric = False
        self.beta_min = 0.1
        self.beta_max = 20.0
        self.num_channels_dae = 64
        self.n_mlp = 3
        self.ch_mult = [1, 1, 2, 2] 
        self.num_res_blocks = 2
        self.attn_resolutions = (16,)
        self.dropout = 0.0
        self.resamp_with_conv = True
        self.conditional = True
        self.fir = True
        self.fir_kernel = [1, 3, 3, 1]
        self.skip_rescale = True
        self.resblock_type = 'biggan'
        self.progressive = 'none'
        self.progressive_input = 'residual'
        self.progressive_combine = 'sum'
        self.embedding_type = 'positional'
        self.fourier_scale = 16.0
        self.not_use_tanh = False
        self.exp = 'SynDiff'
        self.input_path = None 
        self.output_path = None 
        self.dataset = 'cifar10'
        self.image_size = 224
        self.nz = 100
        self.num_timesteps = 4
        self.z_emb_dim = 256
        self.t_emb_dim = 256
        self.batch_size = 1
        self.lr_g = 1.5e-4
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.contrast1 = 'CBCT'
        self.contrast2 = 'CT'
        self.which_epoch = 40
        self.gpu_chose = 0
        self.source = 'CT'