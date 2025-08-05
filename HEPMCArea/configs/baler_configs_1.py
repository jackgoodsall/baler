
# === Configuration options ===

def set_config(c):
    c.input_path                   = "workspaces/ATLAS/data/4momenta.npz"
    c.data_dimension               = 1
    c.compression_ratio            = 1.6
    c.apply_normalization          = True
    c.model_name                   = "AE"
    c.model_type                    = "dense"
    c.epochs                       =  1000 
    c.lr                           = 0.001
    c.batch_size                   = 1024
    c.early_stopping               = True
    c.lr_scheduler                 = True




# === Additional configuration options ===

    c.early_stopping_patience      = 100
    c.min_delta                    = 0
    c.lr_scheduler_patience        = 50
    c.custom_norm                  = False
    c.reg_param                    = 0.001
    c.RHO                          = 0.05
    c.test_size                    = 0.1
    # c.number_of_columns            = 24
    # c.latent_space_size            = 12
    c.extra_compression            = False
    c.intermittent_model_saving    = False
    c.intermittent_saving_patience = 100
    c.mse_avg                      = False
    c.mse_sum                      = True
    c.save_error_bounded_deltas = False
    c.emd                          = False
    c.l1                           = True
    c.activation_extraction        = False
    c.deterministic_algorithm      = False
    c.separate_model_saving        = False

