import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

import os
import csv
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
from ocr_dataset import CaptchaDataset
from model1 import DenseNetOCR_Attn as Model1
from model2 import DenseNetOCR_Attn as Model2
from model3 import DenseNetOCR_Attn as Model3
from model4 import DenseNetOCR_Attn as Model4
from model5 import DenseNetOCR_Attn as Model5
from util import collate_fn, decode_with_confidence, creare_opt_scheduler, decode_output, ensemble,process_rnn_with_prior,decide_final_result


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
num_base = len(base_chars)  
image_height = 64
image_width = 200

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',default=64, help='Batch size')
    parser.add_argument('--epochs', default=300, help='Training epochs')
    parser.add_argument('--warmup_epochs', default=30, help='Warmup epochs')
    parser.add_argument('--output_dir', default="./output/", help='Directory to save outputs')
    parser.add_argument('--train_img_dir', default="../train/images/", help='Training images')
    parser.add_argument('--train_label', default="../train/labels.csv", help='Training labels')
    parser.add_argument('--test_img_dir', default="../test/images/", help='Test images')
    parser.add_argument('--submission_file', default="submission.csv", help='Submission file')
    return parser.parse_args()

def load_dataloaders(args):
    all_files = sorted(os.listdir(args.train_img_dir))
    random.seed(42)
    random.shuffle(all_files)
    split = int(0.95 * len(all_files))
    train_files = all_files[:split]
    val_files = all_files[split:]

    transform_train = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dataset = CaptchaDataset(args.train_img_dir, args.train_label, transform_train, train_files, "train", base_chars, num_base)
    val_dataset = CaptchaDataset(args.train_img_dir, args.train_label, transform_val, val_files, "val", base_chars, num_base)
    
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    return train_dataloader, val_dataloader

def load_test_dataloader(args):
    test_files = sorted(os.listdir(args.test_img_dir))
    transform_test = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    test_dataset = CaptchaDataset(args.test_img_dir, transform=transform_test, file_list=test_files, split="test")
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    return test_dataloader

def train(args,train_dataloader,val_dataloader,model,opt,scheduler):
    model_dict = torch.load("./densenet121.pth", map_location='cpu')
    new_state_dict = {}
    for k, v in model_dict.items():
        new_k = k
        new_k = new_k.replace('norm.1', 'norm1')
        new_k = new_k.replace('norm.2', 'norm2')
        new_k = new_k.replace('conv.1', 'conv1')
        new_k = new_k.replace('conv.2', 'conv2')
        new_state_dict[new_k] = v
    model.backbone.load_state_dict(new_state_dict)
    model = model.to(device)

    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    writer = SummaryWriter(log_dir=f"{args.output_dir}/logs")
    
    best = 0
    steps = 0
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, (imgs, tgt, lens) in enumerate(train_dataloader):
            imgs, tgt, lens = imgs.to(device), tgt.to(device), lens.to(device)
            out = model(imgs)
            out = out.permute(1, 0, 2)

            inp_len = torch.full((imgs.size(0),), out.size(0), device=device, dtype=torch.long)
            loss = crit(out, tgt, inp_len, lens)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_train_loss += loss.item()
            if batch_idx % 10 == 0:
                writer.add_scalar('Train/Loss', loss.item(), steps)
            steps += 1

        avg_train_loss = total_train_loss / len(train_dataloader)
        model.eval()
        correct = total = 0
        val_loss_sum = 0
        with torch.no_grad():
            for batch_idx,(imgs, tgt, lens) in enumerate(val_dataloader):
                imgs, tgt, lens = imgs.to(device), tgt.to(device), lens.to(device)
                out = model(imgs)
                out_permute = out.permute(1, 0, 2)
                inp_len = torch.full((imgs.size(0),), out_permute.size(0), device=device, dtype=torch.long)
                
                v_loss = crit(out_permute, tgt, inp_len, lens)
                val_loss_sum += v_loss.item()

                preds = out.argmax(2).cpu().numpy()
                start = 0
                for i in range(imgs.size(0)):
                    true_ids = tgt[start:start+lens[i]]
                    start += lens[i]

                    true_red = "".join(base_chars[t-1] for t in true_ids if t <= num_base)
                    pred_red = decode_output(preds[i])

                    if true_red == pred_red:
                        correct += 1
                    total += 1

        avg_val_loss = val_loss_sum / len(val_dataloader)
        acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs} | Lr: {opt.param_groups[0]['lr']:.6f} | Train Loss: {avg_train_loss:.4f} | Val Acc: {acc:.2%}",flush=True)
        writer.add_scalar('Train/LR', opt.param_groups[0]['lr'], epoch)
        writer.add_scalar('Val/Accuracy', acc, epoch)
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_epoch.pth"))
            print("Saved best checkpoint.", flush=True)

        torch.save(model.state_dict(), os.path.join(args.output_dir, "last.pth"))
        scheduler.step()
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"{epoch+1},Lr: {opt.param_groups[0]['lr']:.6f},Train loss: {avg_train_loss:.4f},Val acc{acc:.4f}\n")

    print("Best Val Acc:", best)



def predict_vote(args, models, test_dataloader):
    for m in models:
        m.eval()

    res = []
    log_rows = []

    with torch.no_grad():
        for imgs, names in test_dataloader:
            imgs = imgs.to(device)

            probs = [m(imgs).exp() for m in models]

            batch_size = imgs.size(0)
            for i in range(batch_size):
                results = []
                rnn_results = []
                encoder_results = []
                for idx,p in enumerate(probs):
                    s, c = decode_with_confidence(p[i], base_chars, num_base)
                    if idx == 3 or idx == 5:
                        rnn_results.append((s, c)) 
                    elif idx ==1 or idx ==6:
                        encoder_results.append((s, c))
                    else:
                        results.append((s, c))
                s,c = process_rnn_with_prior(rnn_results)
                results.append((s,c))
                s,c = decide_final_result(encoder_results)
                results.append((s,c))
                final_str = ensemble(results)
                need_log = False
                if need_log:
                    row = [names[i]]
                    for s, c in results:
                        row.extend([s, c])
                    log_rows.append(row)

                res.append([names[i], final_str])

    with open(os.path.join(args.output_dir, args.submission_file), "w", newline="") as f:
        csv.writer(f).writerows([["id", "label"]] + res)


if __name__ == "__main__":
    args = args_parser()
    os.makedirs(args.output_dir, exist_ok=True)
    # train_dataloader, val_dataloader = load_dataloaders(args)

    # model1 = Model1(num_base * 2).to(device)
    # opt1,scheduler1 = creare_opt_scheduler(args,model1)
    # train(args, train_dataloader, val_dataloader, model1, opt1, scheduler1)

    # args.batch_size = 32 
    # model2 = Model2(num_base * 2).to(device)
    # opt2,scheduler2 = creare_opt_scheduler(args,model2,lr1=2.5e-4,lr2=8e-4,lr3=1e-5,lr4=1.5e-3,eta_min=5e-5)
    # train(args, train_dataloader, val_dataloader, model2, opt2, scheduler2)

    # args.batch_size = 64
    # args.epochs = 100
    # args.warmup_epochs = 10
    # model3 = Model3(num_base * 2).to(device)
    # opt3,scheduler3 = creare_opt_scheduler(args,model3,lr1=3e-4,lr2=1e-3,lr3=0,lr4=2e-3,eta_min=1e-5)
    # train(args, train_dataloader, val_dataloader, model3, opt3, scheduler3)

    # args.batch_size = 32
    # args.epochs = 300
    # args.warmup_epochs = 30
    # model4 = Model4(num_base * 2).to(device)
    # opt4,scheduler4 = creare_opt_scheduler(args,model4,lr1=3e-4,lr2=1e-4,lr3=1e-4,lr4=2e-3,eta_min=1e-4)
    # train(args, train_dataloader, val_dataloader, model4, opt4, scheduler4)

    # model5 = Model5(num_base * 2).to(device)
    # opt5,scheduler5 = creare_opt_scheduler(args,model5,lr1=3e-4,lr2=1e-4,lr3=1e-5,lr4=2e-3,eta_min=1e-4)
    # train(args, train_dataloader, val_dataloader, model5, opt5, scheduler5)


    test_dataloader = load_test_dataloader(args)
    predict_model1 = Model1(num_base * 2).to(device)
    predict_model1.load_state_dict(torch.load("checkpoints/model1.pth"))
    predict_model2 = Model2(num_base * 2).to(device)
    predict_model2.load_state_dict(torch.load("checkpoints/model2.pth"))
    predict_model3 = Model3(num_base * 2).to(device)
    predict_model3.load_state_dict(torch.load("checkpoints/model3.pth"))
    predict_model4 = Model4(num_base * 2).to(device)
    predict_model4.load_state_dict(torch.load("checkpoints/model4.pth"))
    predict_model5 = Model5(num_base * 2).to(device)
    predict_model5.load_state_dict(torch.load("checkpoints_1/model5.pth"))
    predict_model6 = Model4(num_base * 2).to(device)
    predict_model6.load_state_dict(torch.load("checkpoints_1/model4.pth"))
    predict_model7 = Model2(num_base * 2).to(device)
    predict_model7.load_state_dict(torch.load("./checkpoints_1/model1_bs128.pth"))
    predict_vote(args,[predict_model1,predict_model2,predict_model3,predict_model4,predict_model5,predict_model6,predict_model7],test_dataloader)
