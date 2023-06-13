import importlib
import argparse
import gc
import math
import os
import toml
from multiprocessing import Value

from tqdm import tqdm
import torch
from accelerate.utils import set_seed
import diffusers
from diffusers import DDPMScheduler

import library.train_util as train_util
import library.huggingface_util as huggingface_util
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import apply_snr_weight, pyramid_noise_like
from XTI_hijack import unet_forward_XTI, downblock_forward_XTI, upblock_forward_XTI

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


def train(args):
    if args.output_name is None:
        args.output_name = args.token_string
    use_template = args.use_object_template or args.use_style_template

    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)

    if args.sample_every_n_steps is not None or args.sample_every_n_epochs is not None:
        print(
            "sample_every_n_steps and sample_every_n_epochs are not supported in this script currently / sample_every_n_steps...sample_every_n_epochs..."
        )

    cache_latents = args.cache_latents

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = train_util.load_tokenizer(args)

    # accelerator...
    print("prepare accelerator")
    accelerator, unwrap_model = train_util.prepare_accelerator(args)

    # mixed precision...cast...
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # ...
    text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)

    # Convert the init_word to token_id
    if args.init_word is not None:
        init_token_ids = tokenizer.encode(args.init_word, add_special_tokens=False)
        if len(init_token_ids) > 1 and len(init_token_ids) != args.num_vectors_per_token:
            print(
                f"token length for init words is not same to num_vectors_per_token, init words is repeated or truncated / ...num_vectors_per_token...: length {len(init_token_ids)}"
            )
    else:
        init_token_ids = None

    # add new word to tokenizer, count is num_vectors_per_token
    token_strings = [args.token_string] + [f"{args.token_string}{i+1}" for i in range(args.num_vectors_per_token - 1)]
    num_added_tokens = tokenizer.add_tokens(token_strings)
    assert (
        num_added_tokens == args.num_vectors_per_token
    ), f"tokenizer has same word to token string. please use another one / ...args.token_string...: {args.token_string}"

    token_ids = tokenizer.convert_tokens_to_ids(token_strings)
    print(f"tokens are added: {token_ids}")
    assert min(token_ids) == token_ids[0] and token_ids[-1] == token_ids[0] + len(token_ids) - 1, f"token ids is not ordered"
    assert len(tokenizer) - 1 == token_ids[-1], f"token ids is not end of tokenize: {len(tokenizer)}"

    token_strings_XTI = []
    XTI_layers = [
        "IN01",
        "IN02",
        "IN04",
        "IN05",
        "IN07",
        "IN08",
        "MID",
        "OUT03",
        "OUT04",
        "OUT05",
        "OUT06",
        "OUT07",
        "OUT08",
        "OUT09",
        "OUT10",
        "OUT11",
    ]
    for layer_name in XTI_layers:
        token_strings_XTI += [f"{t}_{layer_name}" for t in token_strings]

    tokenizer.add_tokens(token_strings_XTI)
    token_ids_XTI = tokenizer.convert_tokens_to_ids(token_strings_XTI)
    print(f"tokens are added (XTI): {token_ids_XTI}")
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    if init_token_ids is not None:
        for i, token_id in enumerate(token_ids_XTI):
            token_embeds[token_id] = token_embeds[init_token_ids[(i // 16) % len(init_token_ids)]]
            # print(token_id, token_embeds[token_id].mean(), token_embeds[token_id].min())

    # load weights
    if args.weights is not None:
        embeddings = load_weights(args.weights)
        assert len(token_ids) == len(
            embeddings
        ), f"num_vectors_per_token is mismatch for weights / ...num_vectors_per_token...: {len(embeddings)}"
        # print(token_ids, embeddings.size())
        for token_id, embedding in zip(token_ids_XTI, embeddings):
            token_embeds[token_id] = embedding
            # print(token_id, token_embeds[token_id].mean(), token_embeds[token_id].min())
        print(f"weighs loaded")

    print(f"create embeddings for {args.num_vectors_per_token} tokens, for {args.token_string}")

    # ...
    blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False))
    if args.dataset_config is not None:
        print(f"Load dataset config from {args.dataset_config}")
        user_config = config_util.load_user_config(args.dataset_config)
        ignored = ["train_data_dir", "reg_data_dir", "in_json"]
        if any(getattr(args, attr) is not None for attr in ignored):
            print(
                "ignore following options because config file is found: {0} / ...: {0}".format(
                    ", ".join(ignored)
                )
            )
    else:
        use_dreambooth_method = args.in_json is None
        if use_dreambooth_method:
            print("Use DreamBooth method.")
            user_config = {
                "datasets": [
                    {"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir, args.reg_data_dir)}
                ]
            }
        else:
            print("Train with captions.")
            user_config = {
                "datasets": [
                    {
                        "subsets": [
                            {
                                "image_dir": args.train_data_dir,
                                "metadata_file": args.in_json,
                            }
                        ]
                    }
                ]
            }

    blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
    train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    train_dataset_group.enable_XTI(XTI_layers, token_strings=token_strings)
    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)

    # make captions: tokenstring tokenstring1 tokenstring2 ...tokenstringn ...
    if use_template:
        print("use template for training captions. is object: {args.use_object_template}")
        templates = imagenet_templates_small if args.use_object_template else imagenet_style_templates_small
        replace_to = " ".join(token_strings)
        captions = []
        for tmpl in templates:
            captions.append(tmpl.format(replace_to))
        train_dataset_group.add_replacement("", captions)

        if args.num_vectors_per_token > 1:
            prompt_replacement = (args.token_string, replace_to)
        else:
            prompt_replacement = None
    else:
        if args.num_vectors_per_token > 1:
            replace_to = " ".join(token_strings)
            train_dataset_group.add_replacement(args.token_string, replace_to)
            prompt_replacement = (args.token_string, replace_to)
        else:
            prompt_replacement = None

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, show_input_ids=True)
        return
    if len(train_dataset_group) == 0:
        print("No data found. Please verify arguments / ...")
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latent...color_aug...random_crop..."

    # ... xformers ... memory efficient attention ...
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers)
    diffusers.models.UNet2DConditionModel.forward = unet_forward_XTI
    diffusers.models.unet_2d_blocks.CrossAttnDownBlock2D.forward = downblock_forward_XTI
    diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D.forward = upblock_forward_XTI

    # ...
    if cache_latents:
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
        vae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        accelerator.wait_for_everyone()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    # ...
    print("prepare optimizer, data loader etc.")
    trainable_params = text_encoder.get_input_embeddings().parameters()
    _, _, optimizer = train_util.get_optimizer(args, trainable_params)

    # dataloader...
    # DataLoader...0...
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1 ...
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collater,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # ...
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        print(f"override steps. steps for {args.max_train_epochs} epochs is / ...: {args.max_train_steps}")

    # ...
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr scheduler...
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # accelerator...
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # transform DDP after prepare
    text_encoder, unet = train_util.transform_if_model_is_DDP(text_encoder, unet)

    index_no_updates = torch.arange(len(tokenizer)) < token_ids_XTI[0]
    # print(len(index_no_updates), torch.sum(index_no_updates))
    orig_embeds_params = unwrap_model(text_encoder).get_input_embeddings().weight.data.detach().clone()

    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.requires_grad_(True)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    # text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    if args.gradient_checkpointing:  # according to TI example in Diffusers, train is required
        unet.train()
    else:
        unet.eval()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)

    # ...fp16...PyTorch...fp16...grad scale...
    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)
        text_encoder.to(weight_dtype)

    # resume...
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # epoch...
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # ...
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    print("running training / ...")
    print(f"  num train images * repeats / ...: {train_dataset_group.num_train_images}")
    print(f"  num reg images / ...: {train_dataset_group.num_reg_images}")
    print(f"  num batches per epoch / 1epoch...: {len(train_dataloader)}")
    print(f"  num epochs / epoch...: {num_train_epochs}")
    print(f"  batch size per device / ...: {args.train_batch_size}")
    print(f"  total train batch size (with parallel & distributed & accumulation) / ...: {total_batch_size}")
    print(f"  gradient ccumulation steps / ... = {args.gradient_accumulation_steps}")
    print(f"  total optimization steps / ...: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion" if args.log_tracker_name is None else args.log_tracker_name)

    # function for saving/removing
    def save_model(ckpt_name, embs, steps, epoch_no, force_sync_upload=False):
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        print(f"saving checkpoint: {ckpt_file}")
        save_weights(ckpt_file, embs, save_dtype)
        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

    def remove_model(old_ckpt_name):
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            print(f"removing old checkpoint: {old_ckpt_file}")
            os.remove(old_ckpt_file)

    # training loop
    for epoch in range(num_train_epochs):
        print(f"epoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        text_encoder.train()

        loss_total = 0

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            with accelerator.accumulate(text_encoder):
                with torch.no_grad():
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device)
                    else:
                        # latent...
                        latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215
                b_size = latents.shape[0]

                # Get the text embedding for conditioning
                input_ids = batch["input_ids"].to(accelerator.device)
                # weight_dtype) use float instead of fp16/bf16 because text encoder is float
                encoder_hidden_states = torch.stack(
                    [
                        train_util.get_hidden_states(args, s, tokenizer, text_encoder, weight_dtype)
                        for s in torch.split(input_ids, 1, dim=1)
                    ]
                )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents, device=latents.device)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
                elif args.multires_noise_iterations:
                    noise = pyramid_noise_like(noise, latents.device, args.multires_noise_iterations, args.multires_noise_discount)

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict the noise residual
                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                if args.v_parameterization:
                    # v-parameterization training
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                loss = loss.mean([1, 2, 3])

                if args.min_snr_gamma:
                    loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)

                loss_weights = batch["loss_weights"]  # ...sample...weight
                loss = loss * loss_weights

                loss = loss.mean()  # ...batch_size...

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = text_encoder.get_input_embeddings().parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Let's make sure we don't update any embedding weights besides the newly added token
                with torch.no_grad():
                    unwrap_model(text_encoder).get_input_embeddings().weight[index_no_updates] = orig_embeds_params[
                        index_no_updates
                    ]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # TODO: fix sample_images
                # train_util.sample_images(
                #     accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet, prompt_replacement
                # )

                # ...
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        updated_embs = unwrap_model(text_encoder).get_input_embeddings().weight[token_ids_XTI].data.detach().clone()

                        ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                        save_model(ckpt_name, updated_embs, global_step, epoch)

                        if args.save_state:
                            train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                        remove_step_no = train_util.get_remove_step_no(args, global_step)
                        if remove_step_no is not None:
                            remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                            remove_model(remove_ckpt_name)

            current_loss = loss.detach().item()
            if args.logging_dir is not None:
                logs = {"loss": current_loss, "lr": float(lr_scheduler.get_last_lr()[0])}
                if args.optimizer_type.lower() == "DAdaptation".lower():  # tracking d*lr value
                    logs["lr/d*lr"] = (
                        lr_scheduler.optimizers[0].param_groups[0]["d"] * lr_scheduler.optimizers[0].param_groups[0]["lr"]
                    )
                accelerator.log(logs, step=global_step)

            loss_total += current_loss
            avr_loss = loss_total / (step + 1)
            logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_total / len(train_dataloader)}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        updated_embs = unwrap_model(text_encoder).get_input_embeddings().weight[token_ids_XTI].data.detach().clone()

        if args.save_every_n_epochs is not None:
            saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
            if accelerator.is_main_process and saving:
                ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                save_model(ckpt_name, updated_embs, epoch + 1, global_step)

                remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                if remove_epoch_no is not None:
                    remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                    remove_model(remove_ckpt_name)

                if args.save_state:
                    train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

        # TODO: fix sample_images
        # train_util.sample_images(
        #     accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet, prompt_replacement
        # )

        # end of epoch

    is_main_process = accelerator.is_main_process
    if is_main_process:
        text_encoder = unwrap_model(text_encoder)

    accelerator.end_training()

    if args.save_state and is_main_process:
        train_util.save_state_on_train_end(args, accelerator)

    updated_embs = text_encoder.get_input_embeddings().weight[token_ids_XTI].data.detach().clone()

    del accelerator  # ...

    if is_main_process:
        ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
        save_model(ckpt_name, updated_embs, global_step, num_train_epochs, force_sync_upload=True)

        print("model saved.")


def save_weights(file, updated_embs, save_dtype):
    updated_embs = updated_embs.reshape(16, -1, updated_embs.shape[-1])
    updated_embs = updated_embs.chunk(16)
    XTI_layers = [
        "IN01",
        "IN02",
        "IN04",
        "IN05",
        "IN07",
        "IN08",
        "MID",
        "OUT03",
        "OUT04",
        "OUT05",
        "OUT06",
        "OUT07",
        "OUT08",
        "OUT09",
        "OUT10",
        "OUT11",
    ]
    state_dict = {}
    for i, layer_name in enumerate(XTI_layers):
        state_dict[layer_name] = updated_embs[i].squeeze(0).detach().clone().to("cpu").to(save_dtype)

    # if save_dtype is not None:
    #     for key in list(state_dict.keys()):
    #         v = state_dict[key]
    #         v = v.detach().clone().to("cpu").to(save_dtype)
    #         state_dict[key] = v

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, file)
    else:
        torch.save(state_dict, file)  # can be loaded in Web UI


def load_weights(file):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file

        data = load_file(file)
    else:
        raise ValueError(f"NOT XTI: {file}")

    if len(data.values()) != 16:
        raise ValueError(f"NOT XTI: {file}")

    emb = torch.concat([x for x in data.values()])

    return emb


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, False)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser, False)

    parser.add_argument(
        "--save_model_as",
        type=str,
        default="pt",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .pt) / ...pt...",
    )

    parser.add_argument("--weights", type=str, default=None, help="embedding weights to initialize / ...")
    parser.add_argument(
        "--num_vectors_per_token", type=int, default=1, help="number of vectors per token / ...embeddings..."
    )
    parser.add_argument(
        "--token_string",
        type=str,
        default=None,
        help="token string used in training, must not exist in tokenizer / ...tokenizer...",
    )
    parser.add_argument("--init_word", type=str, default=None, help="words to initialize vector / ...")
    parser.add_argument(
        "--use_object_template",
        action="store_true",
        help="ignore caption and use default templates for object / ...",
    )
    parser.add_argument(
        "--use_style_template",
        action="store_true",
        help="ignore caption and use default templates for stype / ...",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    train(args)
