import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from samrt_adv import SmartPerturbation


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=1., emb_name='model.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='model.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

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
        val_dataloader=None,
        epochs=10,
        scheduler=None
):

    # Tracking best validation accuracy
    best_F1 = 0
    stop_time = 0
    fgm = FGM(model)
    # ema = EMA(model, 0.999)
    # ema.register()
    # smart_attker = SmartPerturbation()
    # Start training loop
    print("Start training...\n")
    print("-" * 60)

    for epoch_i in range(epochs):
        t0_epoch = time.time()
        total_loss = 0
        adv_loss = 0

        model = model.train()

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            output, output1 = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            mix_loss = 1.0 * loss_fn[0](output, targets[:, 0]) + 0.0 * loss_fn[1](output1, targets[:, 1:].float())

            # mix_loss = loss_fn[1](output1, targets[:, 1:].float())
            # mix_loss = 0.6 * loss_fn[0](output, targets[:, 0])
            # task2_loss = []
            # for i, deal in enumerate(torch.argmax(output, dim=1)):
            #     if deal == 1:
            #         task2_loss.append(loss_fn[1](output1[i], targets[i, 1:].float()))
            #
            # if len(task2_loss) > 0:
            #     mix_loss += 0.03 * sum(task2_loss)/len(task2_loss)
            total_loss += mix_loss.item()
            mix_loss.backward()
            # FMG attack
            fgm.attack()  # 在embedding上添加对抗扰动
            loss_adv1, loss_adv2 = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss_adv = 1.0 * loss_fn[0](loss_adv1, targets[:, 0])
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数

            # SMART attack
            # adv_loss, emb_val, eff_perturb = smart_attker.forward(model, output, input_ids, attention_mask)
            # mix_loss += adv_loss
            # adv_loss += adv_loss.item()


            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # ema.update()
            optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        avg_adv_loss = adv_loss/len(train_dataloader)
        # evaluation
        print('avg_adv_loss:  ', avg_adv_loss)
        if val_dataloader is not None:
            val_loss, f1 = evaluate(model, val_dataloader, device, loss_fn, None)
            # val_loss, f1 = evaluate(model, val_dataloader, device, loss_fn)

            # Track the best accuracy
            if f1 >= best_F1:
                best_F1 = f1
                torch.save(model.state_dict(), 'best_model_state.bin')
                stop_time = 0
            else:
                stop_time += 1
                print('reach early stop', stop_time)
                if stop_time > 3:
                    print(f"Training complete! Best F1: {best_F1:.2f}%.")
                    return
            print([epoch_i + 1, avg_train_loss,val_loss, f1])

        torch.save(model.state_dict(), 'last.bin')
        print("\n")
        # print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")
        print(best_F1)


def evaluate(model, val_dataloader, device, loss_fn, ema=None):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()
    # ema.apply_shadow()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    l1 = []
    pred_all = []
    pred2_all = []
    lable_all = []
    lable2_all = []
    output2_all = []
    # For each batch in our validation set...
    with torch.no_grad():
        for batch in val_dataloader:
            # Load batch to GPU
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Compute logits

            output, output1 = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Compute loss
            task1_loss = 1 * loss_fn[0](output, targets[:, 0])
            # task2_loss = []
            # for i, deal in enumerate(targets[:, 0]):
            #     if deal == 1:
            #         task2_loss.append(loss_fn[1](output1[i], targets[i, 1:].float()))

            mix_loss = task1_loss
            # if len(task2_loss) > 0:
            #     task2_loss = 0.2 * sum(task2_loss) / len(task2_loss)
            #     mix_loss += task2_loss
            #     l2.append(task2_loss.item())

            l1.append(task1_loss.item())

            val_loss.append(mix_loss.item())

            # Calculate the accuracy rate
            preds = torch.argmax(output, dim=1).cpu().numpy()
            # preds2 = output1.max(dim=1)[0].cpu().numpy()
            # preds2 = preds2 > 0.3
            target1 = targets[:, 0].cpu().numpy()

            pred_all.extend(preds)
            # pred2_all.extend(preds2)
            lable_all.extend(target1)
            # output2_all.extend(output1.cpu().numpy())
            # lable2_all.extend(targets[:, 1:].cpu().numpy())



        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        # ema.restore()

        for i in range(len(output2_all)):
            output2_all[i] = (output2_all[i] > 0.2).astype(int)

        print('Task1', classification_report(lable_all, pred_all, zero_division=0))
        # print('Task2', classification_report(lable2_all, output2_all, zero_division=0))
    return val_loss, f1_score(lable_all,pred_all, pos_label=1)

def pridict(model, test_dataloader, device):
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    logits = []
    pred_all = []
    mtc = []
    # For each batch in our validation set...
    with torch.no_grad():
        for batch in test_dataloader:
            # Load batch to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Compute logits

            output, output1 = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Calculate the accuracy rate
            logits.extend(F.softmax(output,dim=1).cpu().numpy())
            preds = torch.argmax(output, dim=1).cpu().numpy()
            pred_all.extend(preds)
            # mtc.extend(output1.cpu().numpy())
    return pred_all, logits
