# train a miniature character-level shakespeare model

out_dir = 'out-lichess-uci-fixlr'
eval_interval = 2000 # keep frequent because we'll overfit
eval_iters = 100 # number of batch iterations over which we eval
log_interval = 200 # don't print too too often
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'

always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'chess-gpt-batch'
wandb_run_name = 'lichess_uci_all_elos_8layers_low_batchsize'
wandb_resume = False

dataset = 'lichess_uci_hf_dataset'
gradient_accumulation_steps = 10 # this must be evenly divisiable by ddp world size
batch_size = 20 # using much smaller batch size to see how it affects training
block_size = 1023 # context of up to block_size previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 3e-4#2.26e-4# 3e-4 * batch_size / 50. # with baby networks can afford to go a bit higher
max_iters = 600_000
lr_decay_iters = 600_000 # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10 usually (orig: 3e-5)
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

warmup_iters = 0 # not super necessary potentially
compile = True