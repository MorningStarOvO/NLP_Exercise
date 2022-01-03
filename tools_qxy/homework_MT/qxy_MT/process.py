"""
    本代码用于: 定义「训练过程」和「测试过程」
    创建时间: 2021 年 12 月 27 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 1 月 1 日
"""
# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import os 

# ----- 模型创建相关的 ----- #
import torch
import sacrebleu

# ----- 可视化运行过程 ----- #
from tqdm import tqdm

# ----- 命令行交互相关 ----- # 
import logging

# ----- 导入自定义的包 ----- #
from dataset import chinese_tokenizer_load
from model import beam_search

# ==================== 设置常量参数 ==================== #


# ==================== 函数实现 ==================== #
# ----- 训练 1 EPOCH ----- #
def run_epoch(dataloader, model, criterion, optimizer=None, scheduler=None):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(dataloader):
        # ----- 前馈传播，计算损失 ----- #
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = criterion(out.contiguous().view(-1, out.size(-1)),
                              batch.trg_y.contiguous().view(-1)) / batch.ntokens
        
        # ----- 反向梯度传播 ----- #
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        # ----- 计算 loss ----- #
        total_loss += loss.item() * batch.ntokens.float()
        total_tokens += batch.ntokens
    return total_loss / total_tokens


# ----- 训练进程的实现 ----- #
def train(train_dataloader, dev_dataloader, model, criterion, optimizer, scheduler, local_rank, args):
    global_rank = args.node_rank * args.n_gpu + local_rank
    if local_rank == 0:
        logging.info("------ Start Training! ------")
    best_bleu_score = 0.0
    loss = []
    
    # ----- 是否继续训练 ----- #
    if args.continue_training:
        checkpoint = torch.load(args.model_path)
        epoch_start = checkpoint["epoch"]
        loss = checkpoint["loss"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        logging.info(f"Model at {args.model_path} Loaded. Continue Training from epoch {epoch_start} to {args.epoch_num}")
    else:
        epoch_start = 1

    # ----- 开始训练 ----- #
    for epoch in range(epoch_start, args.epoch_num + 1):
        logging.info(f"[Epoch {epoch}/ Rank {global_rank}] Trainging...")
        model.train()
        train_loss = run_epoch(train_dataloader, model, criterion, optimizer, scheduler)
        loss.append(train_loss)
        logging.info(f'Epoch: {epoch:2d}, loss: {train_loss:.3f}')
        logging.info(f"[Epoch {epoch}] Validating...")

        # ----- 每隔 5 轮进行测试 ----- #
        if epoch % 5 == 0 and epoch > 0:
            evaluate(dev_dataloader, model, local_rank, mode='dev', args=args)

            if global_rank == 0:
                with torch.no_grad():
                    # Load all dev temp results
                    src = []
                    trg = []
                    res = []
                    for i in range(args.n_node * args.n_gpu):
                        result = torch.load(os.path.join(args.temp_dir,
                                                f"dev_rank_{i}"))
                        assert i == result["global_rank"], f"Loading dev results on global rank {i} cause error!"
                        src.extend(result["src"])
                        trg.extend(result["trg"])
                        res.extend(result["res"])
                    bleu_score = sacrebleu.corpus_bleu(res, [trg], tokenize='zh')
                    logging.info(f'Epoch: {epoch:2d} Dev Bleu Score: {bleu_score}')
                    torch.save({
                        "epoch": epoch,
                        "loss": loss,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    }, args.model_path)
                    logging.info(f"[Epoch {epoch}] Module saved!")
                    if float(bleu_score.score) > best_bleu_score:
                        best_bleu_score = float(bleu_score.score)
                        torch.save({
                            "epoch": epoch,
                            "loss": loss,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict()
                        }, args.model_path_best)
                        logging.info(f"Best Module in [Epoch {epoch}] saved!")


# ----- 用训练好的模型预测，保存翻译结果 ----- #
def evaluate(data, model, local_rank, mode, args):
    sp_chn = chinese_tokenizer_load()
    src = []
    trg = []
    res = []
    with torch.no_grad(): 
        for batch in tqdm(data):
            cn_sent = batch.trg_text    # Chinese text
            en_text = batch.src_text    # English text
            src_mask = (batch.src != 0).unsqueeze(-2)
            decode_result, _ = beam_search(model, batch.src, src_mask, args.max_len,
                                            args.padding_idx, args.bos_idx, args.eos_idx,
                                            args.beam_size, local_rank)
            decode_result = [h[0] for h in decode_result]

            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            src.extend(en_text)
            trg.extend(cn_sent)
            res.extend(translation)
    temppath = os.path.join(args.temp_dir, f"{mode}_rank_{args.node_rank * args.n_gpu + local_rank}")
    torch.save({
        "global_rank": args.node_rank * args.n_gpu + local_rank,
        "src": src,
        "trg": trg,
        "res": res
    }, temppath)
    logging.info(f"{mode} dataset evaluation finished, saved at {temppath}")

# ----- 测试模型 ----- #
def test(test_dataloader, model, local_rank, args):
    global_rank = args.node_rank * args.n_gpu + local_rank
    with torch.no_grad():
        checkpoint = torch.load(args.model_path_best)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        logging.info(f"Rank {global_rank}: Model at {args.model_path_best} Loaded. Start testing...")
        evaluate(test_dataloader, model, local_rank, mode='test', args=args)

        if global_rank == 0:
            with torch.no_grad():
                # Load all dev temp results
                src = []
                trg = []
                res = []
                for i in range(args.n_node * args.n_gpu):
                    result = torch.load(os.path.join(args.temp_dir,
                                             f"test_rank_{i}"))
                    assert i == result["global_rank"], f"Loading test results on global rank {i} cause error!"
                    src.extend(result["src"])
                    trg.extend(result["trg"])
                    res.extend(result["res"])
                bleu_score = sacrebleu.corpus_bleu(res, [trg], tokenize='zh')
                logging.info(f'Test Bleu Score: {bleu_score}')
                with open(args.output_path, "w") as fp:
                    for i in range(len(trg)):
                        fp.write(f"[{i}]English sentence: {src[i].strip()}\n")
                        fp.write(f"Translation: {res[i]}\n")
                        fp.write(f"References:  {trg[i]}\n")
