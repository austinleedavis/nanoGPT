# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'lichess_uci_all_elos_8layers'
eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 100 # don't print too too often
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'chess-gpt-batch'
wandb_run_name = 'lichess_uci_all_elos_8layers'

dataset = 'lichess_uci_hf_dataset'
gradient_accumulation_steps = 1
batch_size = 400
block_size = 1023 # context of up to block_size previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 3e-4 # with baby networks can afford to go a bit higher
max_iters = 6e5
lr_decay_iters = 6e5 # make equal to max_iters usually
min_lr = 3e-5 # learning_rate / 10 usually
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

warmup_iters = 10000 # not super necessary potentially
compile = True

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
