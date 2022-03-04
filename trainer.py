import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from adv_model import SmartPerturbation, FGM
from tqdm import tqdm
from loss_function import SymKlCriterion


# https://zhuanlan.zhihu.com/p/68748778
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train_epoch(
        model,
        optimizer,
        device,
        loss_fn,
        train_dataloader,
        args,
        val_dataloader=None,
        epochs=10,
        scheduler=None,
        model_id=''
):

    # Tracking best validation accuracy
    best_F1 = 0
    stop_time = 0
    if args.atk == 'FGM':
        attacker = FGM(model)
    elif args.atk == 'SMART':
        attacker = SmartPerturbation()

    if args.ema:
        ema = EMA(model, 0.999)
        ema.register()

    # Start training loop
    print("Start training...\n")
    print("-" * 60)

    for epoch_i in range(epochs):
        total_loss = 0
        total_adv_loss = 0
        total_kl_loss = 0

        model = model.train()

        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            output, begin_logic,  end_logic = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                out_mtl_drop=True
            )

            mix_loss = 0
            for (weight, fn) in loss_fn:
                mix_loss = mix_loss + weight * fn(output, targets[:, 0])

            if args.r_drop:
                kl_loss = SymKlCriterion(begin_logic, end_logic, alpha=0.5)
                total_kl_loss += kl_loss.item() * 3
                mix_loss = mix_loss + kl_loss * 3

            total_loss += mix_loss.item()

            if args.atk == 'FGM':
                mix_loss.backward()

                # FMG attack
                attacker.attack()  # attack embedding layer
                adv_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                loss_adv = 0
                for (weight, fn) in loss_fn:
                    loss_adv = loss_adv + weight * fn(adv_out, targets[:, 0])

                total_adv_loss += loss_adv.item()
                loss_adv.backward()  # add adv loss to total loss
                attacker.restore()  # restore embedding
            elif args.atk == 'SMART':
                # SMART attack
                adv_loss = attacker.forward(model, output, input_ids, attention_mask)
                mix_loss = mix_loss + adv_loss
                total_adv_loss += adv_loss.item()
                mix_loss.backward()
            else:
                mix_loss.backward()

            if args.grad_clamp:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if args.ema:
                ema.update()
            optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        avg_adv_loss = total_adv_loss/len(train_dataloader)
        avg_kl_loss = total_kl_loss/len(train_dataloader)
        # evaluation
        print('avg_adv_loss:  ', avg_adv_loss)
        print('avg_kl_loss:  ', avg_kl_loss)
        print('train_loss:  ', avg_train_loss)
        if val_dataloader is not None:
            if args.ema:
                val_loss, f1 = evaluate(model, val_dataloader, device, loss_fn, ema)
            else:
                val_loss, f1 = evaluate(model, val_dataloader, device, loss_fn)

            # Track the best accuracy
            if f1 > best_F1:
                best_F1 = f1
                if args.ema:
                    ema.apply_shadow()
                    torch.save(model.state_dict(), model_id+'best_model_state.bin')
                    ema.restore()
                else:
                    torch.save(model.state_dict(), model_id + 'best_model_state.bin')
                stop_time = 0
            else:
                stop_time += 1
                print('reach early stop', stop_time)
                if stop_time > 3:
                    print(f"Training complete! Best F1: {best_F1:.2f}%.")
                    return
            print([epoch_i + 1, avg_train_loss,val_loss, f1])

        print("\n")
        print(best_F1)


def evaluate(model, val_dataloader, device, loss_fn, ema=None):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()
    if ema is not None:
        ema.apply_shadow()

    # Tracking variables
    val_loss = []
    l1 = []
    pred_all = []
    lable_all = []
    # For each batch in our validation set...
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            # Load batch to GPU
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Compute logits

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Compute loss
            task1_loss = 0
            for (weight, fn) in loss_fn:
                task1_loss = task1_loss + weight * fn(output, targets[:, 0])

            mix_loss = task1_loss

            l1.append(task1_loss.item())

            val_loss.append(mix_loss.item())

            # Calculate the accuracy rate
            preds = torch.argmax(output, dim=1).cpu().numpy()
            target1 = targets[:, 0].cpu().numpy()

            pred_all.extend(preds)
            lable_all.extend(target1)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        if ema is not None:
            ema.restore()

        print('Task1', classification_report(lable_all, pred_all, zero_division=0, digits=4))
    return val_loss, f1_score(lable_all,pred_all, pos_label=1)


def pridict(model, test_dataloader, device):
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    logits = []
    pred_all = []
    # For each batch in our validation set...
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # Load batch to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Compute logits

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Calculate the accuracy rate
            logits.extend(F.softmax(output,dim=1).cpu().numpy())
            preds = torch.argmax(output, dim=1).cpu().numpy()
            pred_all.extend(preds)
    return pred_all, logits
