# -*- coding:utf-8 -*-
import os
import sys
import torch
import torch.nn.functional as F

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            feature, target = torch.cat([batch.title,batch.content],dim = 1), batch.label
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            try:
                optimizer.zero_grad()
                logits = model(feature)
                loss = F.cross_entropy(logits, target)
            
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                        '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,loss.item(),train_acc,corrects,batch.batch_size))
            
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        save(model, args.save_dir, 'model_params')
                else:
                    # 加载上一数据
                    # model.load_state_dict(torch.load(os.path.join(args.save_dir,'model_params.pkl')))
                    if steps - last_step >= args.early_stopping:
                        print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                        raise KeyboardInterrupt

def eval(data_iter, model, args):
    # 释放cuda 缓存
    torch.cuda.empty_cache()
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = torch.cat([batch.title,batch.content],dim =1), batch.label
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        try:
            with torch.no_grad(): # 不需要反向传播梯度，所以节约内存
                logits = model(feature)
                loss = F.cross_entropy(logits, target)
            avg_loss += loss.item()
            corrects += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
            else:
                raise e

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,accuracy,corrects,size))
    return accuracy

def save(model, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}.pkl'.format(save_prefix)
    torch.save(model.state_dict(), save_path)
