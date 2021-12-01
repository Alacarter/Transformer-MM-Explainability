import torch
# import CLIP.clip as clip_orig # OpenAI/CLIP
from clip import clip # open_clip repo
from clip.model import * # open_clip repo
from training.main import convert_models_to_fp32 # open_clip repo
import torch.distributed as dist

import PIL
from PIL import Image, ImageDraw
from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import time

MODEL_CONFIGS_DIR = "/scratch/cluster/albertyu/dev/open_clip/src/training/model_configs"

def saliency_map(images, texts, model, preprocess, device, im_size):
    def create_single_saliency_map(index, logits_per_image, images, im_size):
        # one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
        one_hot = np.zeros((images.shape[0], logits_per_image.size()[-1]), dtype=np.float32)
        # one_hot[0, index] = 1
        one_hot[index, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)
        model.zero_grad()
        one_hot.backward(retain_graph=True)

        image_attn_blocks = list(dict(model.module.visual.transformer.resblocks.named_children()).values())
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        for blk in image_attn_blocks:
            grad = blk.attn_grad
            cam = blk.attn_probs
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            R += torch.matmul(cam, R)
        R[0, 0] = 0
        image_relevance = R[0, 1:]
        image_relevance = image_relevance.reshape(1, 1, 7, 7)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=im_size, mode='bilinear')
        image_relevance = image_relevance.reshape(im_size, im_size).cuda().data.cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        image = images[index].permute(1, 2, 0).data.cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        pil_image = Image.fromarray(np.uint8(255 * image))
        image = pil_image.resize((im_size, im_size), resample=PIL.Image.BICUBIC)
        image = np.float32(np.array(image)) / 255
        attn_dot_im = np.tile(np.expand_dims(image_relevance, axis=-1), (1, 1, 3)) * image
        return attn_dot_im, image_relevance, image

    images = torch.cat([preprocess(Image.fromarray(im)).unsqueeze(0).to(device) for im in images])
    # image = preprocess(image).unsqueeze(0).to(device)
    texts = clip.tokenize(texts).to(device)
    image_features, text_features, logit_scale = model(images, texts)
    # cosine similarity as logits
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy() # doesn't seem to be used??

    attn_dot_ims = []
    image_relevances = []
    processed_images = []
    for index in range(images.shape[0]):
        attn_dot_im, image_relevance, image = create_single_saliency_map(index, logits_per_image, images, im_size)
        attn_dot_ims.append(attn_dot_im)
        image_relevances.append(image_relevance)
        processed_images.append(image)

    return np.array(attn_dot_ims), np.array(image_relevances), np.array(processed_images)


def save_heatmap(images, texts, model, preprocess, device, output_fnames, im_size=224):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        img = np.float32(img)
        cam = heatmap + img
        cam = cam / np.max(cam)
        return cam, img

    attn_dot_ims, attns, images = saliency_map(images, texts, model, preprocess, device, im_size=im_size)

    for i in range(len(images)):
        attn_dot_im, attn, image = attn_dot_ims[i], attns[i], images[i]
        output_fname = output_fnames[i]
        text = texts[i]
        vis, orig_img = show_cam_on_image(image, attn)
        vis, orig_img = np.uint8(255 * vis), np.uint8(255 * orig_img)
        attn_dot_im = np.uint8(255 * attn_dot_im)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        vis_concat = np.concatenate([orig_img, vis, attn_dot_im], axis=1)
        vis_concat_captioned = add_caption_to_np_img(vis_concat, text)
        plt.imsave(output_fname, vis_concat_captioned)


def add_caption_to_np_img(im_arr, caption):
    caption_img = Image.new('RGB', (im_arr.shape[1], 20), (255, 255, 255))
    # PIL.Image seems to operate on transposed axes
    d = ImageDraw.Draw(caption_img)
    d.text((5, 5), caption, fill=(0, 0, 0))
    caption_img_arr = np.uint8(np.array(caption_img))
    final_arr = np.concatenate([im_arr, caption_img_arr], axis=0)
    return final_arr


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def load_eval_ids_and_file_id_to_annotation_map(args):
    with open(args.val_file, "r") as f:
        lines = f.readlines()
        val_ex = []
        for line in lines:
            val_ex.append(int(line.rstrip().split(".jpg")[0]))

    with open(args.test_file, "r") as f:
        lines = f.readlines()
        test_ex = []
        for line in lines:
            test_ex.append(int(line.rstrip().split(".jpg")[0]))

    with open(args.annotations_path, "r") as f:
        lines = f.readlines()
        file_id_to_annotation_map = {} # int: str
        for example in lines:
            filename, annotation = example.split("\t")
            file_id = int(filename.split(".jpg")[0]) # removes the .jpg
            if file_id in test_ex:
                file_id_to_annotation_map[file_id] = annotation.rstrip()

    np.random.seed(0)
    file_ids_to_eval = np.random.choice(list(file_id_to_annotation_map.keys()), args.num_evals)
    wall_only_file_ids = [key for key in file_id_to_annotation_map if file_id_to_annotation_map[key] == "wall"]
    wall_pair_file_ids = [key for key in file_id_to_annotation_map
        if ("wall" in file_id_to_annotation_map[key]) and
        (len(file_id_to_annotation_map[key].split(" ")) == 2)
    ]
    file_ids_to_eval = np.concatenate((file_ids_to_eval, np.array(wall_only_file_ids), np.array(wall_pair_file_ids)))
    print("file_ids_to_eval", file_ids_to_eval)
    return file_ids_to_eval, file_id_to_annotation_map


def load_model_preprocess(checkpoint, gpu=0, device="cuda", freeze_clip=True):
    model_class = "ViT-B/32"

    if checkpoint:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:6100",
            world_size=torch.cuda.device_count(),
            rank=gpu,
        )
        model_config_file = os.path.join(MODEL_CONFIGS_DIR, f"{model_class.replace('/', '-')}.json")
        print('Loading model from', model_config_file)
        assert os.path.exists(model_config_file)
        with open(model_config_file, 'r') as f:
            model_info = json.load(f)
        model = CLIP(**model_info)
        convert_weights(model)
        preprocess = clip._transform(model.visual.input_resolution, is_train=False, color_jitter=False)
        convert_models_to_fp32(model)
        model.cuda(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

        checkpoint = torch.load(checkpoint, map_location=device)
        sd = checkpoint["state_dict"]
        model.load_state_dict(sd)

        if freeze_clip:
            print("Freezing clip")
            model.eval()
        else:
            print("Allowing clip to be finetuneable")
            model.train()
    else:
        model, preprocess = clip.load(model_class, device=device, jit=False)

    return model, preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-evals", type=int, required=True)
    # Below: Saliency-like filepaths.
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--annotations-path", type=str, required=True)

    # open_clip-trained checkpoint
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output_heatmaps")
    parser.add_argument("--im-size", type=int, default=224)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_model_preprocess(args.checkpoint, args.gpu)

    file_ids_to_eval, file_id_to_annotation_map = load_eval_ids_and_file_id_to_annotation_map(args)

    # file_ids_to_eval = file_ids_to_eval[:2]
    texts = [file_id_to_annotation_map[eval_id] for eval_id in file_ids_to_eval]
    image_paths = [os.path.join(args.image_dir, f"{eval_id}.jpg") for eval_id in file_ids_to_eval]
    images = np.array([np.array(Image.open(image_path)) for image_path in image_paths])
    output_fnames = [os.path.join(args.output_dir, f"{eval_id}.jpg") for eval_id in file_ids_to_eval]
    save_heatmap(images, texts, model, preprocess, device, output_fnames, args.im_size)

    # attn_dot_im, attn, image = saliency_map(image, text, model, preprocess, device, args.im_size)
    # print("attn_dot_im", attn_dot_im.shape)
    # print("attn", attn.shape)
    # print("image", image.shape)


    # for eval_id in tqdm(file_ids_to_eval):
    #     caption = file_id_to_annotation_map[eval_id]
    #     image_path = os.path.join(args.image_dir, f"{eval_id}.jpg")
    #     image_arr = np.array(Image.open(image_path))
    #     # print(color.BOLD + color.PURPLE + color.UNDERLINE + 'text: ' + texts[0] + color.END)
    #     output_fname = os.path.join(args.output_dir, f"{eval_id}.jpg")
    #     save_heatmap(image_arr, caption, model, preprocess, device, output_fname, args.im_size)

    # produce local copy commands
    commands = []
    for eval_id in file_ids_to_eval:
        output_path = os.path.join(os.getcwd(), args.output_dir)
        command = "scp titan1:{}/{}.jpg .".format(output_path, eval_id)
        commands.append(command)
    print("Copy commands")
    print(commands)
