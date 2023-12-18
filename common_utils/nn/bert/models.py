import numpy as np
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import torch
from torch.optim import Adam, SGD, AdamW
from transformers import get_linear_schedule_with_warmup

def freeze(module):
    """
    Freezes module's parameters.
    """

    for parameter in module.parameters():
        parameter.requires_grad = False


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        outputs, _ = last_hidden_state.max(1)
        return outputs

class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()

        self.meanpool = MeanPooling()
        self.maxpool = MaxPooling()

    def forward(self, last_hidden_state, attention_mask):
        outputs1 = self.meanpool(last_hidden_state, attention_mask)
        outputs2 = self.maxpool(last_hidden_state, attention_mask)
        outputs = torch.cat([outputs1, outputs2], 1)
        return outputs

class LayerPooler(nn.Module):
    def __init__(self, num_hidden_layers, pool_type):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        print('weight layer pool num', self.num_hidden_layers)

        if pool_type == 'last':  # 下段
            weights_init = torch.zeros(self.num_hidden_layers).float()
            weights_init.data[:-1] = -3  # -3
        elif pool_type == 'first':  # 上段
            weights_init = torch.zeros(self.num_hidden_layers).float()
            weights_init.data[1:] = -1  # -3
        elif pool_type == 'mean':  # 中段
            weights_init = torch.zeros(self.num_hidden_layers).float()
            # weights_init.data[:-1] = -3  # -3

        self.layer_weights = torch.nn.Parameter(weights_init)

    def forward(self, x):
        x = (torch.softmax(self.layer_weights, dim=0).unsqueeze(
            1).unsqueeze(1).unsqueeze(1) * x).sum(0)
        return x

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg

        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                cfg.model_first_name + cfg.model_name)

            # regression
            if self.cfg.remove_dropout:
                self.config.hidden_dropout = 0.
                self.config.hidden_dropout_prob = 0.
                self.config.attention_dropout = 0.
                self.config.attention_probs_dropout_prob = 0.

        else:
            self.config = torch.load(config_path)

        if pretrained:
            """
            self.model = AutoModel.from_pretrained(
                cfg.model_first_name + cfg.model_name)
            """
            self.model = AutoModel.from_pretrained(
                cfg.model_first_name + cfg.model_name, config=self.config)

            # print(self.model)

            if cfg.freezing:

                has_embd = hasattr(self.model, 'embeddings')
                if has_embd:
                    print('freeze embd')
                    freeze(self.model.embeddings)
                else:
                    print('freeze shared')
                    freeze(self.model.shared)

                print('freeze layer')

                has_layer = hasattr(self.model.encoder, 'layer')
                print(f'has_layer: {has_layer}')
                if has_layer:
                    print('layer len', len(self.model.encoder.layer))
                    # print(self.model.encoder.layer[:cfg.freezing_count])
                    freeze(self.model.encoder.layer[:cfg.freezing_count])
                elif hasattr(self.model.encoder, 'blocks'):
                    print('blocks len', len(self.model.encoder.blocks))
                    print(self.model.encoder.blocks)
                    freeze(self.model.encoder.blocks[:cfg.freezing_count])
                else:
                    print('freeze encoder')
                    freeze(self.model.encoder)

            if cfg.use_gradient_checkpoint:
                self.model.gradient_checkpointing_enable()

            if cfg.use_reinit:
                print('reinit layer num', cfg.reinit_layers)
                self.reinit_weights()
        else:
            self.model = AutoModel.from_config(self.config)

        if self.cfg.use_attention_head:
            self.attention_head = nn.Sequential(
                nn.Linear(self.config.hidden_size, 512),
                nn.Tanh(),
                nn.Linear(512, 1),
                nn.Softmax(dim=1)
            )
            self._init_weight(self.attention_head)

        self.n_features = self.config.hidden_size

        self.pooler = MeanPooling()

        if self.cfg.output_hidden_states:
            self.n_features *= 4

        self.fc = nn.Sequential(
            nn.Linear(self.n_features, self.cfg.target_size)
        )

        self.dropouts = nn.ModuleList(
            [nn.Dropout(i / 10) for i in range(1, 6)])

    def reinit_weights(self):
        for i in range(self.cfg.reinit_layers):
            self.model.encoder.layer[-(1 + i)].apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, inputs):
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        out = self.model(input_ids=ids, attention_mask=mask,
                         output_hidden_states=self.cfg.output_hidden_states)

        if self.cfg.output_hidden_states:
            out = torch.cat([out["hidden_states"][-1 * i]
                            for i in range(1, 5)], dim=2)
        else:
            out = out.last_hidden_state

        if self.cfg.use_attention_head:
            weights = self.attention_head(out)
            out = torch.sum(weights * out, dim=1)
        else:
            out = self.pooler(out, mask)

        outputs = sum([self.fc(dropout(out))
                       for dropout in self.dropouts]) / 5

        return outputs


class AWP:
    """
    code
    https://github.com/antmachineintelligence/Feedback_1st/blob/main/utils/utils.py
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=0.0001,  # 0.0001
        adv_eps=0.01,  # 0.01
        start_epoch=4,
        adv_step=1,
        scaler=None,
        criterion=None,
        use_amp=True
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler
        self.criterion = criterion
        self.use_amp = use_amp

    def attack_backward(self, x, y, epoch):
        if (self.adv_lr == 0) or (epoch + 1 < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast(self.use_amp):
                # adv_loss, tr_logits = self.model(input_ids=x, attention_mask=attention_mask, labels=y)
                y_preds = self.model(x)

                if y_preds.size(1) == 1:
                    y_preds = y_preds.view(-1)

                adv_loss = self.criterion(y_preds, y)

                adv_loss = adv_loss.mean()

            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class FGM:
    """
    code
    https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/143764
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=0.0001,  # 0.0001
        adv_eps=0.01,  # 0.01
        start_epoch=-4,
        adv_step=1,
        scaler=None,
        criterion=None,
        use_amp=True
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        # self.backup_eps = {}
        self.scaler = scaler
        self.criterion = criterion
        self.use_amp = use_amp

    def attack_backward(self, x, y, epoch):
        if (self.adv_lr == 0) or (epoch + 1 < self.start_epoch):
            return None

        # self._save()

        self.attack()
        with torch.cuda.amp.autocast(self.use_amp):
            # adv_loss, tr_logits = self.model(input_ids=x, attention_mask=attention_mask, labels=y)
            y_preds = self.model(x)

            if y_preds.size(1) == 1:
                y_preds = y_preds.view(-1)

            adv_loss = self.criterion(y_preds, y)

            adv_loss = adv_loss.mean()

        # self.optimizer.zero_grad()
        self.scaler.scale(adv_loss).backward()

        self.restore()

    def attack(self, epsilon=0.0001, emb_name='word_embeddings'):  # epsilon=0.1
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # print(name)
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}

    """
    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
    """


def get_optimizer(model, cfg):
    if cfg.use_lr_decay:
        return get_optimizer_lr_decay(model, cfg.lr, lr_decay=cfg.lr_decay, change_head_lr=cfg.change_head_lr)
    else:
        return get_optimizer_default(model, cfg.lr)

def get_optimizer_default(model, lr):
    # optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.1},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(0.9, 0.98), lr=lr)

    return optimizer

def get_optimizer_lr_decay(model, lr, lr_decay=0.9, weight_decay=0.1, change_head_lr=False):

    model_type = 'model'
    no_decay = ["bias", "LayerNorm.weight"]
    head_lr = lr

    if change_head_lr:
        print('change_head_lr')
        head_lr *= 10

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if 'lstm' in n
                       or 'cnn' in n
                       or 'fc' in n
                       or 'attention_head' in n],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    num_layers = model.config.num_hidden_layers

    """
    layers = list(getattr(model, model_type).encoder.layer) + \
        [getattr(model, model_type).embeddings]
    """
    has_layer = hasattr(getattr(model, model_type).encoder, 'layer')
    print(f'has_layer: {has_layer}')
    if has_layer:
        layers = [getattr(model, model_type).embeddings] + \
            list(getattr(model, model_type).encoder.layer)
    else:
        layers = [getattr(model, model_type).embeddings] + \
            list(getattr(model, model_type).encoder.blocks)

    layers.reverse()

    for layer in layers:
        # print(layer)
        lr *= lr_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    print('min lr', lr)

    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(0.9, 0.98), lr=lr)

    return optimizer


def get_scheduler_warmup(train_loader, optimizer, epochs, num_warmup_steps_rate):
    num_train_optimization_steps = int(len(train_loader) * epochs)
    num_warmup_steps = int(
        num_train_optimization_steps * num_warmup_steps_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    return scheduler
