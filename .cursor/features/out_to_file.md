# Output to File

I would like for the training program in the current repo to be able to optionally output its results to a data file. This will allow for better recordkeeping of the various outputs as users experiment with the models.

Each run of `training.py` should generate one output file. The output files should have the following components:

## File path

User should be able to specify the file path for the output file. Default should be `/outputs` (relative path from the repo root). 

## File name

The output file name should include the following information:
- Project name (`build_llm_output`)
- Model name (`bigram` is currently the only one, I might add more. The program needs to be able to detect the model name to add to the file name.)
- The name of the data source used (see files under `build_llm_karpathy/sources`), WITHOUT the file extension. Example: `carolyn_hax_103125_chat` 
- Vocab size used in the run
- Number of training steps
- Whether it was run in test mode or not
- Static file name, in app caps (`OUTPUT`)
- Timestamp: `MMDDYYY_HHMMSS`

- Snake case should be used as a delimiter

So one example full file name would be: `/outputs/build_llm_output_bigram_carolyn_hax_103125_chat_65_1000_test=true_OUTPUT_10312025_101534`

## File content

The file itself should contain the following:

- The hyperparameters used. Example: 
```python
    batch_size = 32
    block_size = 64
    training_steps = 1000
    eval_interval = 100
    learning_rate = 3e-4
    eval_iters = 50
    n_embd = 128
    n_head = 4
    n_layer = 3
    dropout = 0.2
```
- All of the terminal/stdout output. Example from a recent run:
```sh
🔬 TEST MODE: Using reduced hyperparameters for fast training
✅ Using Apple Silicon GPU (Metal Performance Shaders)
Device: mps
Model size: 3 layers, 128 embedding dims, 4 heads
[00:00:00] Pre-processing sequences       ██████ 0        /        0[00:00:00] Tokenize words                 ██████ 1738     /     1738
[00:00:00] Count pairs                    ██████ 1738     /     1738
[00:00:00] Compute merges                 ██████ 271      /      271
✅ Custom BPE tokenizer trained (vocab_size=360)
Total model parameters: 694,632
Model device: mps:0
✅ Model successfully moved to Apple Silicon GPU (MPS)
ℹ️  Using MPS without compilation (torch.compile disabled for MPS)
Starting training for 1000 steps...
Batch size: 32, Block size: 64
Vocabulary size: 360 tokens
--------------------------------------------------
step 0/1000 (0.0%): train loss 6.0638, val loss 6.0451 | 0.7s (0.00 steps/sec)
step 25/1000 (2.5%) | 17.91 steps/sec | ETA: 0.9m
step 50/1000 (5.0%) | 26.34 steps/sec | ETA: 0.6m
step 75/1000 (7.5%) | 31.26 steps/sec | ETA: 0.5m
step 100/1000 (10.0%): train loss 5.0566, val loss 5.1347 | 3.4s (29.62 steps/sec)
step 125/1000 (12.5%) | 32.22 steps/sec | ETA: 0.5m
step 150/1000 (15.0%) | 34.25 steps/sec | ETA: 0.4m
step 175/1000 (17.5%) | 35.89 steps/sec | ETA: 0.4m
step 200/1000 (20.0%): train loss 4.4167, val loss 4.5754 | 5.9s (33.79 steps/sec)
step 225/1000 (22.5%) | 35.05 steps/sec | ETA: 0.4m
step 250/1000 (25.0%) | 36.16 steps/sec | ETA: 0.3m
step 275/1000 (27.5%) | 37.01 steps/sec | ETA: 0.3m
step 300/1000 (30.0%): train loss 4.0007, val loss 4.2761 | 8.5s (35.32 steps/sec)
step 325/1000 (32.5%) | 36.10 steps/sec | ETA: 0.3m
step 350/1000 (35.0%) | 36.84 steps/sec | ETA: 0.3m
step 375/1000 (37.5%) | 37.50 steps/sec | ETA: 0.3m
step 400/1000 (40.0%): train loss 3.7500, val loss 4.1494 | 11.0s (36.41 steps/sec)
step 425/1000 (42.5%) | 36.99 steps/sec | ETA: 0.3m
step 450/1000 (45.0%) | 37.53 steps/sec | ETA: 0.2m
step 475/1000 (47.5%) | 38.04 steps/sec | ETA: 0.2m
step 500/1000 (50.0%): train loss 3.6045, val loss 4.0783 | 13.5s (37.12 steps/sec)
step 525/1000 (52.5%) | 37.57 steps/sec | ETA: 0.2m
step 550/1000 (55.0%) | 38.00 steps/sec | ETA: 0.2m
step 575/1000 (57.5%) | 38.39 steps/sec | ETA: 0.2m
step 600/1000 (60.0%): train loss 3.4672, val loss 4.0739 | 16.0s (37.59 steps/sec)
step 625/1000 (62.5%) | 37.96 steps/sec | ETA: 0.2m
step 650/1000 (65.0%) | 38.25 steps/sec | ETA: 0.2m
step 675/1000 (67.5%) | 38.59 steps/sec | ETA: 0.1m
step 700/1000 (70.0%): train loss 3.3685, val loss 4.0415 | 18.5s (37.75 steps/sec)
step 725/1000 (72.5%) | 38.05 steps/sec | ETA: 0.1m
step 750/1000 (75.0%) | 38.37 steps/sec | ETA: 0.1m
step 775/1000 (77.5%) | 38.66 steps/sec | ETA: 0.1m
step 800/1000 (80.0%): train loss 3.2691, val loss 4.0733 | 21.1s (37.94 steps/sec)
step 825/1000 (82.5%) | 38.21 steps/sec | ETA: 0.1m
step 850/1000 (85.0%) | 38.48 steps/sec | ETA: 0.1m
step 875/1000 (87.5%) | 38.74 steps/sec | ETA: 0.1m
step 900/1000 (90.0%): train loss 3.1414, val loss 4.0352 | 23.6s (38.12 steps/sec)
step 925/1000 (92.5%) | 38.35 steps/sec | ETA: 0.0m
step 950/1000 (95.0%) | 38.59 steps/sec | ETA: 0.0m
step 975/1000 (97.5%) | 38.82 steps/sec | ETA: 0.0m
--------------------------------------------------
Training complete! Final loss: 3.1610
Total training time: 25.6s (39.04 steps/sec)

Generating text...
==================================================
x e with out li mo ve le . S y because t res at ment - re n es husband is sa me or n ' t to con re d to ge st about a be a r and $ up in the ir ad to o n ev ue and that ch an other p re st ) and . l it ( wh ic al our one y need to im es d ue in it was re ll h o c li w on g w are of op , but ha v i es k ion s and need ing . I s ! I sh o p s sa me es ri ght s , S ty and s , wh y no mo ve to b le ar ing , we e i z ed that be t ant ag es en se c ate ly go c us s it li ght ay , th ou th ou bi l ity to b work s n has as m un happ it s . Th en from my tr y for , th you ' s i th o you ' re me to ce he ' re I don ' t we want to do I f f act about p ts m un c . Th ate to p a st y that she ev ai l ink ent o se she want ing to ma ke h es . I me to re ct iv your g en . Th an who is w here , as on g iv ity to he ' s it is c ould k in in ter ms to s with some ght ed , but that for t that g ,
==================================================
```

## Other details
If needed, you can create a planning document with the steps needed to undertake this. If you create this, store it in `.cursor/tmp` in the repo.
