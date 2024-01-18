# train a miniature character-level shakespeare model

out_dir = 'out-lichess-400x10'
eval_interval = 4000 # keep frequent because we'll overfit
eval_iters = 100 # number of batch iterations over which we eval
log_interval = 100 # don't print too too often
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
reset_iters = False 

always_save_checkpoint = True

wandb_log = True 
wandb_resume = False
wandb_project = 'chess-gpt-batch'
wandb_run_name = 'lichess-uci-smallbatch20'

dataset = 'lichess_uci_hf_dataset'
gradient_accumulation_steps = 1 # this must be evenly divisiable by ddp world size
batch_size = 320 # using much smaller batch size to see how it affects training

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 768
dropout = 0
block_size = 1024 # d_context

learning_rate = 3e-4
max_iters = 6e6
lr_decay_iters = 6e6 # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10 usually (orig: 3e-5)
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

warmup_iters = 1000 # not super necessary potentially
compile = True