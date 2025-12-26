import torch
import torch.nn as nn
import math
import torch.optim as optim
from collections import defaultdict
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
def collate_fn(batch):
    imgs, targets, lens = zip(*batch)
    return torch.stack(imgs), torch.cat(targets), torch.stack(lens)


def decode_output(sequence, base_chars, num_base):
    res = ""
    last = 0
    for p in sequence:
        if p != last and p != 0 and p <= num_base:
            res += base_chars[p - 1]
        last = p
    return res

def decode_with_confidence(probs_tensor, base_chars, num_base):
    max_probs, indices = probs_tensor.max(dim=1)
    indices = indices.cpu().numpy()
    max_probs = max_probs.cpu().numpy()
    
    res = ""
    last = 0
    conf_scores = [] 
    
    for i, p in enumerate(indices):
        
        if p != last and p != 0 and p <= num_base:
            res += base_chars[p - 1]
            conf_scores.append(max_probs[i]) 
        
        last = p
    
    if len(conf_scores) == 0:
        final_conf = max_probs.mean() 
    else:
        final_conf = sum(conf_scores) / len(conf_scores)
        
    return res, final_conf

def creare_opt_scheduler(args,model,lr1=1e-4,lr2=1e-4,lr3=1e-5,lr4=2e-3,eta_min=1e-5):
    opt = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr1},
        {"params": model.map.parameters(), "lr": lr2},
        {"params": model.attn_encoder.parameters(), "lr": lr3},
        {"params": model.cls.parameters(), "lr": lr4},
    ], weight_decay=1e-4)
    scheduler = SequentialLR(
        opt,
        [
            LinearLR(opt, start_factor=0.1, total_iters=args.warmup_epochs),
            CosineAnnealingLR(opt, T_max=args.epochs - args.warmup_epochs, eta_min=eta_min)
        ],
        milestones=[args.warmup_epochs]
    )
    return opt,scheduler


def decode_with_char_confidence_from_tokens(tokens, logits, base_chars, num_base):
    """
    tokens: List[int] 预测的 token ids (CTC)
    logits: [T, C] 概率矩阵 (softmax 或 log_softmax 后)
    输出:
        chars: 原始字符序列，不 collapse
        conf_scores: 对应每个字符的置信度
    """
    probs = logits.exp().cpu().numpy()  # [T, C]
    chars = []
    conf_scores = []
    last = -1
    for t, token in enumerate(tokens):
        if token != 0 and token <= num_base:
            chars.append(base_chars[token-1])
            conf_scores.append(probs[t, token])
        last = token
    return chars, conf_scores

def ensemble(results):
    """
    results: List of (str, conf), length = 5
    """
    preds = []
    for i in range(5):
        s, c = results[i]
        if isinstance(s, str) and s != "":
            preds.append((i+1,s, c))

    groups = defaultdict(list)
    for s, c in results:
        groups[s].append(c)

    # [(str, count, mean_conf)]
    stats = [
        (s, len(cs), sum(cs) / len(cs))
        for s, cs in groups.items()
    ]

    # 按票数排序
    stats.sort(key=lambda x: x[1], reverse=True)

    top_s, top_cnt, top_conf = stats[0]

    # ===== Case 1: >=4 =====
    if top_cnt >= 3:
        return top_s

    # ===== Case 3: exactly one 2-of-5 =====
    twos = [(s, cnt, mc) for s, cnt, mc in stats if cnt == 2]
    if len(twos) == 1:
        repeated_str = twos[0][0]
        repeated_confs = [c for _, s, c in preds if s == repeated_str]
        repeated_mean_conf = sum(repeated_confs) / len(repeated_confs)
        candidates = []
        candidates.append((repeated_str, repeated_mean_conf))

        for _,s, c in preds:
            if s != repeated_str:
                candidates.append((s, c))

        candidates.sort(key=lambda x: x[1], reverse=True)

        if abs(repeated_confs[0] - repeated_confs[1])  > 0.24:
            selected_str = candidates[1][0]
            return selected_str
        tmp = sorted(stats, key=lambda x: x[2], reverse=True)
        top_s, top_cnt, top_conf = tmp[0]
        if abs(repeated_confs[0] - repeated_confs[1]) > 0.05 and  abs(repeated_confs[0] - repeated_confs[1]) < 0.24 and top_conf - max(repeated_confs) > 0.016:
            return top_s
        return twos[0][0]

    # ===== Case 4: two 2-of-5 =====
    if len(twos) == 2:
        mc1 = twos[0][2]
        mc2 = twos[1][2]
        twos_strs = [t[0] for t in twos]
        s1, _, mc1 = twos[0]
        s2, _, mc2 = twos[1]
        other_confs = [(s, c) for _, s, c in preds if s not in twos_strs]
        other_confs = other_confs[0][1]
        if other_confs > mc1 and other_confs < mc2 or other_confs > mc2 and other_confs < mc1:
            max_mc = max(mc1, mc2)
            min_mc = min(mc1, mc2)
            single_items = [(mid, s, c) for mid, s, c in preds if s not in twos_strs]
            single_mid, single_s, single_conf = single_items[0]
            if single_mid == 5 or max_mc - min_mc < 0.066:
                return single_s
        twos.sort(key=lambda x: x[2], reverse=True)
        return twos[0][0]

    # ===== Case 5: fallback =====
    return max(preds, key=lambda x: x[2])[1]


LOW_CONF_PRIOR = {
    frozenset(("6", "0")),
    frozenset(("O", "Q")),
    frozenset(("B", "R")),
    frozenset(("3", "9")), 
    frozenset(("2", "9")),   
    frozenset(("3", "6")), 
    frozenset(("X", "1")), 
    frozenset(("V", "M")),
    frozenset(("V", "W")),
    frozenset(("0", "O")),
    frozenset(("2", "S")),      
}



def process_rnn_with_prior(rnn_results):
    (s1, c1), (s2, c2) = rnn_results
    if s1 == s2:
        return s1, c1 if c1 - c2 >= 0.2 else c2
    # 高低 conf
    if c1 >= c2:
        high_s, high_c = s1, c1
        low_s, low_c = s2, c2
    else:
        high_s, high_c = s2, c2
        low_s, low_c = s1, c1

    if len(high_s) != len(low_s):
        return high_s, high_c

    for a, b in zip(high_s, low_s):
        if a != b and frozenset((a, b)) in LOW_CONF_PRIOR:
            if high_c - low_c < 0.2: 
                return low_s, low_c
            else:
                return high_s, high_c
    return high_s, high_c

def decide_final_result(attention_results):
    (s1, c1), (s2, c2) = attention_results
    if s1 == s2:
        return s1,c1
    if c1 < 0.90 and c2 > 0.95:
        return s2, c2
        
    if len(s1) == len(s2) and s1 != s2:
        is_only_zl_diff = True
        for char1, char2 in zip(s1, s2):
            if char1 != char2:
                if not ((char1 == 'E' and char2 == 'L') or (char1 == 'L' and char2 == 'E')):
                    is_only_zl_diff = False
                    break
        if is_only_zl_diff:
            return s2, c1
        
    if len(s1) == len(s2):
        new_label = []
        use_latter_count = 0
        
        for char1, char2 in zip(s1, s2):
            if char1 in ['I', 'l'] and char2 == '1':
                new_label.append(char2)
                use_latter_count += 1
                
            elif char1 == '0' and char2 == '6':
                new_label.append(char2)
                use_latter_count += 1
                
            elif char1 == '7' and char2 == 'I':
                new_label.append(char2)
                use_latter_count += 1
                
            else:
                new_label.append(char1) 
        
        if use_latter_count > 0:
            return "".join(new_label), c2

    return s1, c1