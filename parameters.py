parameters = {
    'batch_size': 32,
    'max_epoch': 40,
    'learning_rate': 0.0003,
    'field_emb_dim': 50,  # field embedding
    'pos_embed_dim': 5,  # position embedding
    'word_embed_dim': 400,  # word embedding
    'hidden_dim': 500,  # hidden layer
    'dropout': 0.4,
    'weight_decay': 0.0,
    'grad_clip': 5.0,
    'seed': 1,
    'beam_width': 1,
    'max_len': 60,
    'max_field': 100,
    'pos_size': 31,
    'train_mode': False,  # training (if false, then doing inference)
    'resume': True,  # loading the trained model
    'copy': False  # using copy mechanism
}
