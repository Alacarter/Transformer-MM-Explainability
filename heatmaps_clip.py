import torch
import CLIP.clip as clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse

def interpret(image, text, model, device, output_fname, index=None):
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    if index is None:
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
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

    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    plt.imsave(output_fname, vis)


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
    print("file_ids_to_eval", file_ids_to_eval)
    return file_ids_to_eval, file_id_to_annotation_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, required=False)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--num-evals", type=int, required=True)
    # Below: Saliency-like filepaths.
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--annotations-path", type=str, required=True)

    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output_samples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    file_ids_to_eval, file_id_to_annotation_map = load_eval_ids_and_file_id_to_annotation_map(args)

    for eval_id in file_ids_to_eval:
        caption = file_id_to_annotation_map[eval_id]
        image_path = os.path.join(args.image_dir, f"eval_id".jpg)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize([caption]).to(device)
        # print(color.BOLD + color.PURPLE + color.UNDERLINE + 'text: ' + texts[0] + color.END)
        output_fname = f"{eval_id}".jpg
        interpret(model=model, image=image, text=text, device=device, output_fname=output_fname, index=0)

    # produce local copy commands
    commands = []
    for eval_id in file_ids_to_eval:
        command = "scp titan1:{}/{}.jpg .".format(args.output_dir, eval_id)
        commands.append(command)
    print("Copy commands")
    print(commands)
