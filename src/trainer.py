from abc import ABC, abstractmethod
import torch
import os
from cs336_basics.loaders.model_loader import save_checkpoint
from cs336_basics.train.adamw import AdamW
from cs336_basics.train.cross_entropy import cross_entropy
from argparse import ArgumentParser
from cs336_basics.loaders.data_loader import data_loader
import json
from cs336_basics.language_model.transformer_lm import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.language_model.softmax import softmax
from cs336_basics.train.gradient_clipping import gradient_clipping
from cs336_basics.train.learning_rate_schedule import learning_rate_schedule, WarmupCosine
from cs336_basics.train.contrastive_loss import contrastive_loss

class Trainer(ABC):

    def __init__(self, model, train_input_ids, val_input_ids, data_loader_fn, tokenizer, config):
        self.model = model
        self.train_input_ids = train_input_ids
        self.val_input_ids = val_input_ids
        self.data_loader_fn = data_loader_fn
        self.tokenizer = tokenizer
        self.config = config
        self.val_data = self._prepare_val_input_ids(self.config['batch_size'])
    
    @abstractmethod
    def fit(self):
        pass

class SFTTrainer(Trainer):

    def __init__(self, model, train_input_ids, val_input_ids, data_loader_fn, tokenizer, config):
        super().__init__(model, train_input_ids, val_input_ids, data_loader_fn, tokenizer, config)
        self.loss_fn = cross_entropy
        self.optimizer = AdamW(
            self.model.parameters(),
            config['weight_decay'],
            config['learning_rate']['max_learning_rate'],
            config['betas'],
            config['eps']
        )
        self.lr_scheduler = WarmupCosine(
            self.optimizer, 
            config['learning_rate']['max_learning_rate'], 
            config['learning_rate']['min_learning_rate'], 
            config['learning_rate']['warmup_iters'], 
            config['learning_rate']['cosine_cycle_iters']
        )


    def _data_loader_wrapper(self, train_input_ids, num_samples):
        for _ in range(num_samples):
            input_ids, labels = self.data_loader_fn(train_input_ids, self.config['batch_size'], self.config['context_length'], 'cuda')
            yield {'input_ids': input_ids, 'labels': labels}

    def _prepare_val_input_ids(self, batch_size):
        len_val_input_ids = len(self.val_input_ids)
        cutoff = len_val_input_ids - (len_val_input_ids % (self.config['context_length'] * batch_size))
        val_input_ids = self.val_input_ids[:cutoff].to(self.model.device)
        val_input_ids = val_input_ids.reshape(-1, batch_size, self.config['context_length'])
        input_ids = val_input_ids[:, :, :-1]
        labels = val_input_ids[:, :, 1:]
        return {'input_ids': input_ids, 'labels': labels}

    def prepare_lr_scheduler(self):
        return lambda it: learning_rate_schedule(it, self.config['learning_rate']['max_learning_rate'], self.config['learning_rate']['min_learning_rate'], self.config['learning_rate']['warmup_iters'], self.config['learning_rate']['cosine_cycle_iters'])

    def fit(self):
        global_step = 0
        self.model.train()

        # check all paramters have required gradients
        for param in self.model.parameters():
            assert param.requires_grad, "Parameter does not have required gradients"

        
        for batch in self._data_loader_wrapper(self.train_input_ids, self.config['train_steps']):
            
            metrics = {}
            
            self.optimizer.zero_grad()
            self.model.zero_grad()

            logits, final_hidden_states = self.model(batch['input_ids'], return_hidden_states=True)
            loss = self.loss_fn(logits, batch['labels'])

            metrics['lm_loss'] = loss.item()
            
            if self.config['contrastive_loss']['use_contrastive_loss']:
                c_loss = contrastive_loss(final_hidden_states, self.config['contrastive_loss']['n_samples'])
                beta = self.config['contrastive_loss']['beta']
                loss += beta * c_loss

            loss.backward()
            grad_norm = gradient_clipping(self.model.parameters(), self.config['gradient_clipping'])
            optimizer_metrics = self.optimizer.step()
            self.lr_scheduler.step()

            metrics.update({
                'loss': loss.item(),
                'grad_norm': grad_norm,
                'acc': torch.sum(torch.argmax(logits, dim=-1) == batch['labels']).item() / batch['labels'].numel(),
                'c_loss': c_loss.item()
            })
            
            metrics.update(optimizer_metrics)
            global_step += 1
            
            if global_step % self.config['val_interval'] == 0:
                val_metrics = self.validate()
                metrics.update(val_metrics)
            
            if global_step % self.config['log_interval'] == 0:
                print(f"Step {global_step}, Metrics: {metrics}")

            if global_step % self.config['save_interval'] == 0:
                save_checkpoint(self.model, self.optimizer, global_step, os.path.join(self.config['save_dir'], f"checkpoint_{global_step}.pt"))

    def validate(self):
        self.model.eval()
        metrics = {}
        val_loss = 0
        val_lm_loss = 0
        val_c_loss = 0
        with torch.no_grad():
            for i in range(self.val_data['input_ids'].shape[0]):
                logits, final_hidden_states = self.model(self.val_data['input_ids'][i], return_hidden_states=True)
                loss = self.loss_fn(logits, self.val_data['labels'][i])
                val_loss += loss.item()
                val_lm_loss += loss.item()
                if self.config['contrastive_loss']['use_contrastive_loss']:
                    c_loss = contrastive_loss(final_hidden_states, self.config['contrastive_loss']['n_samples'])
                    val_c_loss += c_loss.item()
                    val_loss += self.config['contrastive_loss']['beta'] * c_loss.item()

            print(f"Generated text: {self.generate(self.tokenizer.decode(self.val_data['input_ids'][0][0].tolist()[:32]), 32)}")
            metrics = {
                'val_loss': val_loss / self.val_data['input_ids'].shape[0],
                'val_acc': torch.sum(torch.argmax(logits, dim=-1) == self.val_data['labels'][i]).item() / self.val_data['labels'][i].numel(),
                'val_lm_loss': val_lm_loss / self.val_data['input_ids'].shape[0],
                'val_c_loss': val_c_loss / self.val_data['input_ids'].shape[0]
            }
        return metrics

    def generate(self, text, max_new_tokens):
        with torch.no_grad():
            input_ids = self.tokenizer.encode(text).unsqueeze(0).to(self.model.device)
            output = self.model.generate(input_ids, max_new_tokens, temperature=1.0, do_sample=True, eos_token=self.tokenizer.eos_token_id, pad_token=self.tokenizer.pad_token_id)
            return self.tokenizer.decode(output[0].tolist())

def pretty_log(log_str: str):
    print(f"\033[92m{log_str}\033[0m")

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()

    config = json.load(open(args.config))


    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        attn_pdrop=config['attn_pdrop'],
        residual_pdrop=config['residual_pdrop']
    ).to('cuda')

    pretty_log(f"model: {model}")
    
    pretty_log("Loading tokenizer")
    if 'pretrained_tokenizer' in config:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['pretrained_tokenizer'])
    else:
        tokenizer = Tokenizer.from_files(config['vocab_filepath'], config['merges_filepath'], config['special_tokens'])

    # print the tokenizer
    print(f"vocab size: {len(tokenizer.vocab)}")

    # check if train_input_ids and val_input_ids exist
    if not os.path.exists(config['train_input_ids']):
        pretty_log("Loading val dataset")
        val_dataset = open(config['val_dataset_path']).read()
        train_dataset = open(config['train_dataset_path']).read()
        pretty_log(f"Encoding train dataset: {len(train_dataset)}")
        train_input_ids = tokenizer.encode(train_dataset)
        pretty_log(f"Encoding val dataset: {len(val_dataset)}")
        val_input_ids = tokenizer.encode(val_dataset)
        torch.save(train_input_ids, config['train_input_ids'])
        torch.save(val_input_ids, config['val_input_ids'])
    else:
        train_input_ids = torch.load(config['train_input_ids'])
        val_input_ids = torch.load(config['val_input_ids'])

    print(f"Train input ids: {len(train_input_ids)}")
    print(f"Val input ids: {len(val_input_ids)}")

    trainer = SFTTrainer(model, train_input_ids, val_input_ids, data_loader, tokenizer, config)
    trainer.fit()

if __name__ == '__main__':
    main()