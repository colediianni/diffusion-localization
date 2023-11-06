import argparse
import numpy as np
import os
import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import LOG_DIR, get_formatstr
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode

import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform


def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)


def eval_prob_adaptive(unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None, vae=None):
    scheduler_config = get_scheduler_config(args)
    T = scheduler_config['num_train_timesteps']
    max_n_samples = max(args.n_samples)

    if all_noise is None:
        all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), device=latent.device)
    if args.dtype == 'float16':
        all_noise = all_noise.half()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

    data = dict()
    t_evaluated = set()
    remaining_prmpt_idxs = list(range(len(text_embeds)))
    start = T // max_n_samples // 2
    t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]

    for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
        ts = []
        noise_idxs = []
        text_embed_idxs = []
        curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
        # print(curr_t_to_eval)
        curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
        for prompt_i in remaining_prmpt_idxs:
            for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                ts.extend([t] * args.n_trials)
                noise_idxs.extend(list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1))))
                text_embed_idxs.extend([prompt_i] * args.n_trials)
        t_evaluated.update(curr_t_to_eval)
        pred_errors = eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                 text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss, vae=vae)
        # match up computed errors to the data
        for prompt_i in remaining_prmpt_idxs:
            mask = torch.tensor(text_embed_idxs) == prompt_i
            prompt_ts = torch.tensor(ts)[mask]
            prompt_pred_errors = pred_errors[mask]
            if prompt_i not in data:
                data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
            else:
                data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
                data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

        # compute the next remaining idxs
        errors = [-data[prompt_i]['pred_errors'].mean() for prompt_i in remaining_prmpt_idxs]
        best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
        remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]

    # organize the output
    assert len(remaining_prmpt_idxs) == 1
    pred_idx = remaining_prmpt_idxs[0]

    return pred_idx, data


def eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
               text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2', vae=None):
    assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
    pred_errors = torch.zeros(len(ts), device='cpu')
    if loss[:3] == "all":
        pred_errors = None
    idx = 0

    with torch.inference_mode():
        for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
            batch_ts = torch.tensor(ts[idx: idx + batch_size])
            noise = all_noise[noise_idxs[idx: idx + batch_size]]
            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                            noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
            # with torch.no_grad():
            #     img = vae.decode(latent/0.18215).sample
            #     img = img.mul_(0.5).add_(0.5).clamp(0, 1).type(torch.DoubleTensor)
            #     print(img.shape)
            #     print(img.min(), img.max())
            #     plt.imshow(img.detach().cpu()[0].permute(1, 2, 0))
            #     plt.show()

            t_input = batch_ts.to(device).half() if dtype == 'float16' else batch_ts.to(device)
            text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]
            noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
            if loss == 'l2':
                error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'l1':
                error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'huber':
                error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == "all_l1":
                true_image = convert_latent_to_img(noised_latent, vae, latent.dtype)
                # error = F.mse_loss(noise, noise_pred, reduction='none')
                predicted_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                            noise_pred * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
                predicted_image = convert_latent_to_img(predicted_latent, vae, latent.dtype)
                error = torch.abs(true_image - predicted_image)

                if pred_errors == None:
                    pred_errors = torch.zeros([len(ts), predicted_image.shape[1], predicted_image.shape[2], predicted_image.shape[3]], device='cpu')
            else:
                raise NotImplementedError
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)
    return pred_errors

def convert_latent_to_img(latent, vae, data_type):
    latent = latent.detach().to(device).type(data_type)
    img = vae.decode(latent/0.18215).sample
    img = img.mul_(0.5).add_(0.5).type(torch.DoubleTensor)
    img = img.detach().cpu()
    return img

def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='pets',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    # run args
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Number of trials per timestep')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--prompt_path', type=str, required=True, help='Path to csv file with prompts to use')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber', "all_l1", "all_l2"), help='Type of loss to use')

    # args for adaptively choosing which classes to continue trying
    parser.add_argument('--to_keep', nargs='+', type=int, required=True)
    parser.add_argument('--n_samples', nargs='+', type=int, required=True)


    parser.add_argument('--localization', type=bool, default=False, help='Whether to do classification or class localization')
    parser.add_argument('--test_file_path', type=str, default="", required=True, help='Path to image file to run localization on')

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    # make run output folder
    name = f"v{args.version}_{args.n_trials}trials_"
    name += '_'.join(map(str, args.to_keep)) + 'keep_'
    name += '_'.join(map(str, args.n_samples)) + 'samples'
    if args.interpolation != 'bicubic':
        name += f'_{args.interpolation}'
    if args.loss == 'l1':
        name += '_l1'
    elif args.loss == 'huber':
        name += '_huber'
    if args.img_size != 512:
        name += f'_{args.img_size}'
    if args.extra is not None:
        run_folder = osp.join(LOG_DIR, args.dataset + '_' + args.extra, name)
    else:
        run_folder = osp.join(LOG_DIR, args.dataset, name)
    if args.test_file_path != "":
        if args.extra is not None:
            run_folder = osp.join(LOG_DIR, 'custom_img_' + args.extra, name)
        else:
            run_folder = osp.join(LOG_DIR, 'custom_img', name)

    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    # set up dataset and prompts
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    latent_size = args.img_size // 8
    if args.test_file_path != "":
        from PIL import Image 
        img = Image.open(args.test_file_path) 
        img = transform(img)
        target_dataset = [[img, 0]]
    else:
        target_dataset = get_target_dataset(args.dataset, train=args.split == 'train', transform=transform)
    prompts_df = pd.read_csv(args.prompt_path)

    # load pretrained models
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    unet.eval()
    vae.eval()
    torch.backends.cudnn.benchmark = True

    # load noise
    if args.noise_path is not None:
        assert not args.zero_noise
        all_noise = torch.load(args.noise_path).to(device)
        print('Loaded noise from', args.noise_path)
    else:
        all_noise = None

    # refer to https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L276
    text_input = tokenizer(prompts_df.prompt.tolist(), padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(text_input.input_ids), 100):
            text_embeddings = text_encoder(
                text_input.input_ids[i: i + 100].to(device),
            )[0]
            embeddings.append(text_embeddings)
    text_embeddings = torch.cat(embeddings, dim=0)
    assert len(text_embeddings) == len(prompts_df)

    if args.localization:
        # idxs_to_eval = list(range(10000))
        idxs_to_eval = list(range(len(target_dataset)))
        pbar = tqdm.tqdm(idxs_to_eval)
        for i in pbar:
            image, label = target_dataset[i]
            # print(label)
            if label != 0:
                continue
            with torch.no_grad():
                img_input = image.to(device).unsqueeze(0)
                if args.dtype == 'float16':
                    img_input = img_input.half()
                x0 = vae.encode(img_input).latent_dist.mean
                x0 *= 0.18215

                # print(x0.shape)
                img = convert_latent_to_img(x0, vae, x0.dtype)
                plt.imshow(img[0].permute(1, 2, 0).clamp(0, 1))
                plt.savefig(osp.join(run_folder, str(i) + '_original_img.png'))
                # plt.show()
                plt.close()
                # stop

                pred_idx, pred_errors = eval_prob_adaptive(unet, x0, text_embeddings, scheduler, args, latent_size, all_noise, vae=vae)

                total_diff = 0
                for j in range(len(pred_errors[label]["pred_errors"])):
                    pos_prediction = pred_errors[label]["pred_errors"][j]
                    labels = list(pred_errors.keys())
                    for k in labels:
                        if k == label:
                            continue
                        neg_prediction = pred_errors[k]["pred_errors"][j]
                        # if torch.sum(pos_prediction) < torch.sum(neg_prediction): # only consider the data point if it is positively contributing to the correct class's prediction
                        total_diff = total_diff + torch.abs(neg_prediction - pos_prediction) # to visualize the pixels contributing most to classificaion
                        # total_diff = total_diff + (neg_prediction - pos_prediction) # to visualize the pixels positively contributing
                if isinstance(total_diff, int):
                    continue
                total_diff = total_diff.permute(1, 2, 0).sum(dim=2)
                total_diff = total_diff + total_diff.min()
                plt.imshow(total_diff / total_diff.max())
                plt.savefig(osp.join(run_folder, str(i) + 'total_localization_img.png'))
                # # plt.show()
                plt.close()

    else:
        # subset of dataset to evaluate
        if args.subset_path is not None:
            idxs = np.load(args.subset_path).tolist()
        else:
            idxs = list(range(len(target_dataset)))
        idxs_to_eval = idxs[args.worker_idx::args.n_workers]

        formatstr = get_formatstr(len(target_dataset) - 1)
        correct = 0
        total = 0
        pbar = tqdm.tqdm(idxs_to_eval)
        for i in pbar:
            if total > 0:
                pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
            fname = osp.join(run_folder, formatstr.format(i) + '.pt')
            if os.path.exists(fname):
                print('Skipping', i)
                if args.load_stats:
                    data = torch.load(fname)
                    correct += int(data['pred'] == data['label'])
                    total += 1
                continue
            image, label = target_dataset[i]
            with torch.no_grad():
                img_input = image.to(device).unsqueeze(0)
                if args.dtype == 'float16':
                    img_input = img_input.half()
                x0 = vae.encode(img_input).latent_dist.mean
                x0 *= 0.18215
            pred_idx, pred_errors = eval_prob_adaptive(unet, x0, text_embeddings, scheduler, args, latent_size, all_noise)
            pred = prompts_df.classidx[pred_idx]
            torch.save(dict(errors=pred_errors, pred=pred, label=label), fname)
            if pred == label:
                correct += 1
            total += 1


if __name__ == '__main__':
    main()
