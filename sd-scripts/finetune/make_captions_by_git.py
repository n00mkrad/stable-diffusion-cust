import argparse
import os
import re

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.generation.utils import GenerationMixin

import library.train_util as train_util


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATTERN_REPLACE = [
    re.compile(r'(has|with|and) the (words?|letters?|name) (" ?[^"]*"|\w+)( ?(is )?(on|in) (the |her |their |him )?\w+)?'),
    re.compile(r'(with a sign )?that says ?(" ?[^"]*"|\w+)( ?on it)?'),
    re.compile(r"(with a sign )?that says ?(' ?(i'm)?[^']*'|\w+)( ?on it)?"),
    re.compile(r"with the number \d+ on (it|\w+ \w+)"),
    re.compile(r'with the words "'),
    re.compile(r"word \w+ on it"),
    re.compile(r"that says the word \w+ on it"),
    re.compile("that says'the word \"( on it)?"),
]

# ... with the word xxxx ...


def remove_words(captions, debug):
    removed_caps = []
    for caption in captions:
        cap = caption
        for pat in PATTERN_REPLACE:
            cap = pat.sub("", cap)
        if debug and cap != caption:
            print(caption)
            print(cap)
        removed_caps.append(cap)
    return removed_caps


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def main(args):
    # GIT...1...: transformers 4.26.0...
    org_prepare_input_ids_for_generation = GenerationMixin._prepare_input_ids_for_generation
    curr_batch_size = [args.batch_size]  # ...batch_size...

    # input_ids...
    # ...
    def _prepare_input_ids_for_generation_patch(self, bos_token_id, encoder_outputs):
        input_ids = org_prepare_input_ids_for_generation(self, bos_token_id, encoder_outputs)
        if input_ids.size()[0] != curr_batch_size[0]:
            input_ids = input_ids.repeat(curr_batch_size[0], 1)
        return input_ids

    GenerationMixin._prepare_input_ids_for_generation = _prepare_input_ids_for_generation_patch

    print(f"load images from {args.train_data_dir}")
    train_data_dir_path = Path(args.train_data_dir)
    image_paths = train_util.glob_images_pathlib(train_data_dir_path, args.recursive)
    print(f"found {len(image_paths)} images.")

    # ...cache...
    print(f"loading GIT: {args.model_id}")
    git_processor = AutoProcessor.from_pretrained(args.model_id)
    git_model = AutoModelForCausalLM.from_pretrained(args.model_id).to(DEVICE)
    print("GIT loaded")

    # captioning...
    def run_batch(path_imgs):
        imgs = [im for _, im in path_imgs]

        curr_batch_size[0] = len(path_imgs)
        inputs = git_processor(images=imgs, return_tensors="pt").to(DEVICE)  # ...pil...
        generated_ids = git_model.generate(pixel_values=inputs.pixel_values, max_length=args.max_length)
        captions = git_processor.batch_decode(generated_ids, skip_special_tokens=True)

        if args.remove_words:
            captions = remove_words(captions, args.debug)

        for (image_path, _), caption in zip(path_imgs, captions):
            with open(os.path.splitext(image_path)[0] + args.caption_extension, "wt", encoding="utf-8") as f:
                f.write(caption + "\n")
                if args.debug:
                    print(image_path, caption)

    # ...DataLoader...
    if args.max_data_loader_n_workers is not None:
        dataset = train_util.ImageLoadingDataset(image_paths)
        data = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    b_imgs = []
    for data_entry in tqdm(data, smoothing=0.0):
        for data in data_entry:
            if data is None:
                continue

            image, image_path = data
            if image is None:
                try:
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                except Exception as e:
                    print(f"Could not load image path / ...: {image_path}, error: {e}")
                    continue

            b_imgs.append((image_path, image))
            if len(b_imgs) >= args.batch_size:
                run_batch(b_imgs)
                b_imgs.clear()

    if len(b_imgs) > 0:
        run_batch(b_imgs)

    print("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / ...")
    parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption file / ...")
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/git-large-textcaps",
        help="model id for GIT in Hugging Face / ...GIT...Hugging Face...ID",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / ...")
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=None,
        help="enable image reading by DataLoader with this number of workers (faster) / DataLoader...",
    )
    parser.add_argument("--max_length", type=int, default=50, help="max length of caption / caption...")
    parser.add_argument(
        "--remove_words",
        action="store_true",
        help="remove like `with the words xxx` from caption / `with the words xxx`...",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--recursive", action="store_true", help="search for images in subfolders recursively / ...")

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
