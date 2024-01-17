# train a miniature character-level shakespeare model

out_dir = 'out-lichess-uci-smallbatch'
eval_interval = 2000 # keep frequent because we'll overfit
eval_iters = 100 # number of batch iterations over which we eval
log_interval = 400 # don't print too too often
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
reset_iters = True 

always_save_checkpoint = True

wandb_log = True 
wandb_resume = False
wandb_project = 'chess-gpt-batch'
wandb_run_name = 'lichess-uci-smallbatch'

dataset = 'lichess_uci_hf_dataset'
gradient_accumulation_steps = 10 # this must be evenly divisiable by ddp world size
batch_size = 2 # using much smaller batch size to see how it affects training

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
block_size = 1023 # context of up to block_size previous characters

learning_rate = 3e-4
max_iters = 2e6
lr_decay_iters = 2e6 # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10 usually (orig: 3e-5)
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

warmup_iters = 1000 # not super necessary potentially
compile = True