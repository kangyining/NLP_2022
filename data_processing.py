from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torch


class DPMDataset(Dataset):
    def __init__(self, dataframe, with_label, tokenizer, max_len):
        self.reviews = dataframe.text.to_numpy()
        self.with_label = with_label
        if with_label:
            self.targets = dataframe.label.to_numpy()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, output_hidden_states=True, return_dict=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        if self.with_label:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(self.targets[item], dtype=torch.long)
            }
        else:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }


def create_data_loader(df, with_label, tokenizer, max_len, batch_size, sampler=None, shuffle=False):
    ds = DPMDataset(
        df,
        with_label,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        sampler=sampler,
        shuffle=shuffle
    )

