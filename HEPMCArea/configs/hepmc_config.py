
# === Configuration options ===

def set_config(c):
    c.input_path                   = "workspaces/ATLAS/data/Eptetaphi_qt_final copy.npz"
    c.data_dimension               = 1
    c.compression_ratio            = 3.125
    c.apply_normalization          = False
    c.model_name                   = "TransformerAE_two"
    c.model_type                    = "dense"
    c.epochs                       =  100
    c.lr                           = 0.00005
    c.batch_size                   = 128
    c.early_stopping               = True   
    c.lr_scheduler                 = True
    c.float_dtype = "float32"


 

# === Additional configuration options ===

    c.early_stopping_patience      = 100
    c.min_delta                    = 0
    c.lr_scheduler_patience        = 20
    c.custom_norm                  = True
    c.reg_param                    = 0
    c.RHO                          = 0.05
    c.test_size                    = 0.1
    c.number_of_columns            = 200
    c.latent_space_size            = 64
    c.extra_compression            = False
    c.intermittent_model_saving    = False
    c.intermittent_saving_patience = 10
    c.mse_avg                      = False
    c.mse_sum                      = True
    c.save_error_bounded_deltas =    False
    c.emd                          = False
    c.l1                           = True
    c.activation_extraction        = False
    c.deterministic_algorithm      = False
    c.separate_model_saving        = False

