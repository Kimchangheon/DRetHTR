import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from DRetHTR.augments.ocrodeg import OcrodegAug
import numpy as np
from tokenizers.processors import TemplateProcessing

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_tgt_length=128, height=96, width=96, do_aug=True):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_tgt_length
        self.width = width
        self.height = height
        self.do_aug = do_aug
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("L")
        pixel_values = self.processor.process_image(
            image, size_dict=(self.height, self.width),
            rescale_factor=0.00392156862745098,
            image_mean=0.5,
            image_std=0.5,
            do_aug=self.do_aug,
            do_resize=True,
            do_rescale=True,
            do_normalize=True
        )
        # add labels (input_ids) by encoding the text
        labels =  self.processor.tokenize(text, self.max_target_length)
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.PAD_ID else -100 for label in labels]
        encoding = {"pixel_values": pixel_values, "labels": torch.tensor(labels)}
        return encoding

class ImageTxtProcessor:
    def __init__(self, char_map_or_tokenizer, p_aug, padding, max_length, bpe=False, gpt2=False):
        self.augs = OcrodegAug(
            p_random_vert_pad=p_aug,
            p_random_hori_pad=p_aug,
            p_random_squeeze_stretch=p_aug,
            p_dilation=p_aug,
            p_erosion=p_aug,
            p_distort_with_noise=p_aug,
            p_background_noise=p_aug,
        )
        self.resize_transform = transforms.Resize
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize

        self.bpe = bpe
        self.gpt2 = gpt2
        if bpe :
            self.tokenizer = char_map_or_tokenizer  # BPE tokenizer

            # automatic padding/truncation can be enabled like below
            # self.tokenizer.enable_truncation(max_length= max_length)
            # self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<PAD>"), pad_token="<PAD>", length= max_length)
            if gpt2 :
                self.BOS_ID = self.tokenizer.bos_token_id
                self.PAD_ID = self.tokenizer.pad_token_id
                self.EOS_ID = self.tokenizer.eos_token_id
                self.UNK_ID = self.tokenizer.unk_token_id
            else :
                # Retrieve special token IDs (optional if you need them for decode logic)
                self.BOS_ID = self.tokenizer.token_to_id("<BOS>")
                self.PAD_ID = self.tokenizer.token_to_id("<PAD>")
                self.EOS_ID = self.tokenizer.token_to_id("<EOS>")
                self.UNK_ID = self.tokenizer.token_to_id("<UNK>")
        else :
            self.char_map = char_map_or_tokenizer
            self.PAD_ID = self.char_map['<PAD>']
        self.padding = padding
        self.max_length = max_length

    def process_image(self, image, size_dict=None,rescale_factor=0.00392156862745098,  image_mean=0.5, image_std=0.5,
                      do_aug=True, do_resize = True, do_rescale= True, do_normalize = True):
        #augmentation
        if do_aug :
            image = self.augs(image)

        # Resize
        # image = np.array(image)
        if self.padding :
            size_dict = (64, int(image.width * (64 / image.height)))
            if do_resize and size_dict is not None:
                resize_transform = self.resize_transform(size_dict)
                image = resize_transform(image)

            # Convert to NumPy array if it's not already
            if not isinstance(image, np.ndarray):
                image = np.array(image)

            # Add padding to reach a width of 2227
            if image.shape[1] < self.max_length :
                padding_width = self.max_length - image.shape[1]
                padding = ((0, 0), (0, padding_width))  # only pad width for a 2D array
                image = np.pad(image, pad_width=padding, mode='constant', constant_values=255)
            else :
                image = image[:,:self.max_length]

        else : # no padding and fixed size(28,420)
            if do_resize and size_dict is not None:
                resize_transform = self.resize_transform(size_dict)
                image = resize_transform(image)


        # Rescale
        # it is already resclaed because of to_tensor_transform
        if do_rescale:
            image = self.to_tensor_transform(image)
        # Inversion
        image = 1-image
        # Normalize
        if do_normalize:
            normalize_transform = self.normalize_transform(mean=image_mean, std=image_std)
            image = normalize_transform(image)

        return image

    def tokenize(self, text, max_length):
        if self.bpe :
            # automatic padding/truncation make the tokenize method simple like below
            # return self.tokenizer.encode(text).ids
            """
            Tokenize a string using the BPE tokenizer.

            1. Encode text to get token IDs.
            2. (Optional) Add BOS and EOS if you did not set a post-processor.
            3. Pad or truncate to `max_length`.
            """
            if self.gpt2:
                # GPT-2 tokenizer from Hugging Face returns a list directly.
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            else:
                # Custom BPE tokenizer (from tokenizers library) returns an Encoding object.
                encoding = self.tokenizer.encode(text)
                token_ids = encoding.ids

            # Add BOS, EOS if your tokenizer is not automatically adding them
            # If you used TemplateProcessing in HF tokenizers, it might do this for you.
            token_ids = [self.BOS_ID] + token_ids + [self.EOS_ID]

            # Pad if shorter than max_length
            if len(token_ids) < max_length:
                token_ids += [self.PAD_ID] * (max_length - len(token_ids))
            # Truncate if longer
            token_ids = token_ids[:max_length]
            return token_ids
        else :
            char_map = self.char_map
            tokenized = [char_map['<BOS>']]
            for char in text:
                tokenized.append(char_map.get(char, char_map['<UNK>']))
            tokenized.append(char_map['<EOS>'])
            tokenized.extend([char_map['<PAD>']] * (max_length - len(tokenized)))
            return tokenized[:max_length]

    # unkown token to ' '
    def decode(self, pred_ids, skip_special_tokens=True):
        if self.bpe :
            """
            Decode a list of token IDs back into a string.
            If skip_special_tokens=True, filter out <BOS>, <EOS>, <PAD>, <UNK>.
            """
            # Convert to Python int if you are dealing with torch tensors
            if hasattr(pred_ids, "detach"):
                pred_ids = pred_ids.detach().cpu().tolist()

            # If it's still nested, handle that accordingly.
            # We'll assume pred_ids is a 1D list for single example decode.
            filtered_ids = []
            for idx in pred_ids:
                if skip_special_tokens and idx in [
                    self.BOS_ID, self.EOS_ID, self.PAD_ID
                ]:
                    continue
                filtered_ids.append(idx)

            # Convert IDs back to string
            decoded_text = self.tokenizer.decode(filtered_ids)
            return decoded_text
        else :
            # Inverting the char_map to map numbers back to characters
            inv_char_map = {v: k for k, v in self.char_map.items()}

            # Decoding the list of numbers back into a string
            decoded = ''
            for id in pred_ids:
                id = id.item()
                if skip_special_tokens and id in [self.char_map['<BOS>'], self.char_map['<EOS>'], self.char_map['<PAD>'],
                                                  self.char_map['<UNK>']]:
                    continue
                decoded += inv_char_map.get(id, '')
            return decoded

    def batch_decode(self, pred_ids_batch, skip_special_tokens=True):
        # Decoding a batch of lists
        return [self.decode(pred_ids, skip_special_tokens) for pred_ids in pred_ids_batch]
