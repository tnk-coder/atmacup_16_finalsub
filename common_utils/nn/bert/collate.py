import torch

class Collate:
    def __init__(self, tokenizer, objective_bert):
        self.tokenizer = tokenizer
        self.objective_bert = objective_bert

    def __call__(self, batch):
        use_label = "label" in batch[0].keys()
        use_token_type_ids = "token_type_ids" in batch[0].keys()
        # print(use_token_type_ids)

        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"]
                                    for sample in batch]
        if use_token_type_ids:
            output["token_type_ids"] = [sample["token_type_ids"]
                                        for sample in batch]
        if use_label:
            output["label"] = [sample["label"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(input_ids) for input_ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [
                s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0]
                                        for s in output["attention_mask"]]
            if use_token_type_ids:
                output["token_type_ids"] = [s + (batch_max - len(s)) * [0]
                                            for s in output["token_type_ids"]]
        else:
            output["input_ids"] = [
                (batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] +
                                        s for s in output["attention_mask"]]
            if use_token_type_ids:
                output["token_type_ids"] = [(batch_max - len(s)) * [0] +
                                            s for s in output["token_type_ids"]]
        # convert to tensors
        output["input_ids"] = torch.tensor(
            output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(
            output["attention_mask"], dtype=torch.long)
        if use_token_type_ids:
            output["token_type_ids"] = torch.tensor(
                output["token_type_ids"], dtype=torch.long)

        if use_label:
            if self.objective_bert == 'multiclass':
                output["label"] = torch.tensor(
                    output["label"], dtype=torch.long)
            else:
                output["label"] = torch.tensor(
                    output["label"], dtype=torch.float)
            return output, output["label"]
        else:
            return output
