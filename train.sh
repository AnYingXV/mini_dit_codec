accelerate launch train.py \
    --clip_path /hdd/u202411103070/projects/img_research/all_checkpoints/CLIP \
    --dit_path /hdd/u202411103070/projects/img_research/all_checkpoints/SANA \
    --elic_path /hdd/u202411103070/projects/img_research/all_checkpoints/ELIC/elic_official.pth \
    \
    --gradient_accumulation_steps 8 \
    --mixed_precision bf16 \
    --seed 42 \
    --report_to tensorboard \
    \
    --train_image_root /hdd/u202411103070/projects/datasets/flickr8k/images/Flicker8k_Dataset \
    --train_caption_file /hdd/u202411103070/projects/datasets/flickr8k/texts/Flickr8k.token.txt \
    --test_dataset /hdd/u202411103070/projects/datasets \
    --train_patch_size 256 \
    --train_batch_size 4 \
    --dataloader_num_workers 8 \
    \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing \
    \
    --max_train_steps 1000 \
    --train_stage 1 \
    --checkpointing_steps 250 \
    --eval_freq 100 \
    \
    --lambda_rate 1.0 \
    --lambda_mse 1.0 \
    --lambda_lpips 1.0 \
    --lambda_dists 1.0 \
    --lambda_distill 1.0 \
    --lambda_cond 1.0 \
    --lambda_adv 0.0 \
    \
    --max_grad_norm 1.0 \
    --output_path /hdd/u202411103070/projects/img_research/mini_dit_codec/train_output
