import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

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
    # Start training loop
    print("Start training...\n")
    print("-" * 60)

    for epoch_i in range(epochs):
        t0_epoch = time.time()
        total_loss = 0
        model = model.train()

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            output, output1 = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            mix_loss = 0.1 * loss_fn[0](output, targets[:, 0]) + 1 * loss_fn[1](output1, targets[:, 1:].float())
            # mix_loss = 0.8 * loss_fn[0](output, targets[:, 0])
            # task2_loss = []
            # for i, deal in enumerate(targets[:, 0]):
            #     if deal == 1:
            #         task2_loss.append(loss_fn[1](output1[i], targets[i, 1:].float()))
            #
            # if len(task2_loss) > 0:
            #     mix_loss += 0.2 * sum(task2_loss)/len(task2_loss)
            total_loss += mix_loss.item()

            mix_loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)

        # evaluation

        if val_dataloader is not None:
            val_loss, val_accuracy, f1, precision, recall = evaluate(model, val_dataloader, device, loss_fn)

            # Track the best accuracy
            if f1 > best_F1:
                best_F1 = f1
                torch.save(model.state_dict(), 'best_model_state.bin')
            else:
                stop_time += 1
                print('reach early stop', stop_time)
                if stop_time > 4:
                    print(f"Training complete! Best F1: {best_F1:.2f}%.")
                    return
            print([epoch_i + 1, avg_train_loss,val_loss, val_accuracy, f1, precision, recall])

        torch.save(model.state_dict(), 'last.bin')
        print("\n")
        # print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")
        print(best_F1)


def evaluate(model, val_dataloader, device, loss_fn):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    l1 = []
    l2 = []
    pred_all = []
    pred2_all = []
    lable_all = []
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
            task1_loss = 0.8 * loss_fn[0](output, targets[:, 0])
            task2_loss = []
            for i, deal in enumerate(targets[:, 0]):
                if deal == 1:
                    task2_loss.append(loss_fn[1](output1[i], targets[i, 1:].float()))

            mix_loss = task1_loss
            if len(task2_loss) > 0:
                task2_loss = 0.2 * sum(task2_loss) / len(task2_loss)
                mix_loss += task2_loss
                l2.append(task2_loss.item())

            l1.append(task1_loss.item())

            # mix_loss = task1_loss + task2_loss

            val_loss.append(mix_loss.item())

            # Calculate the accuracy rate
            preds = torch.argmax(output, dim=1).cpu().numpy()
            preds2 = output1.max(dim=1)[0].cpu().numpy()
            preds2 = preds2 > 0.3
            targets = targets[:, 0].cpu().numpy()
            accuracy = (preds == targets).mean()
            val_accuracy.append(accuracy)


            pred_all.extend(preds)
            pred2_all.extend(preds2)
            lable_all.extend(targets)




    # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy) * 100
        print(np.mean(l1), np.mean(l2), f1_score(lable_all,pred2_all))

    return val_loss, val_accuracy, f1_score(lable_all,pred_all), precision_score(lable_all,pred_all), recall_score(lable_all,pred_all)

def pridict(model, test_dataloader, device):
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
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
            preds = torch.argmax(output, dim=1).cpu().numpy()
            pred_all.extend(preds)
            mtc.append(output1)
    return pred_all, mtc
