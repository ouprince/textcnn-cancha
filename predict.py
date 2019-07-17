# -*- coding:utf-8 -*-
import sys,os
import torch
import model

args = torch.load('snapshot/args.pkl')
text_field = torch.load(os.path.join(args.save_dir,'textfield.pkl'))
label_field = torch.load(os.path.join(args.save_dir,'labelfield.pkl'))
model = model.TextCNN(args)
model.load_state_dict(torch.load(os.path.join(args.save_dir,'model_params.pkl')))
if args.cuda:
    torch.cuda.set_device(args.device)
    model = model.cuda()

model.eval()

def predict(content,title = None):
    if title:
        title = [text_field.preprocess(title)]
        title = text_field.process(title)
    content = [text_field.preprocess(content)]
    content = text_field.process(content)
    
    if title is not None:
        net = torch.cat([title,content],dim = 1)
    else:
        net = content
    
    if args.cuda:
        net = net.cuda()

    logits = model(net)
    logits = torch.max(logits, 1)[1].view(logits.size(0)).tolist()[0]
    return label_field[logits]
    '''
    sentence = [text_field.preprocess(sentence)]
    net = text_field.process(sentence)
    logits = model(net)
    logits = torch.max(logits, 1)[1].view(logits.size(0)).tolist()[0]
    return label_field[logits] 
    '''
if __name__ == "__main__":
    y_pred = []
    y_true = []
    with open("data/shuku.dev.data",encoding = "utf-8") as readme:
        for line in readme:
            label, title, content = line.split("\t")
            y_true.append(label)
            y_pred.append(predict(content = content, title = title))
    from sklearn.metrics import classification_report
    labels = ['-1','0','1']
    target_names = ["负面","中性","正面"]
    print(classification_report(y_true,y_pred,labels = labels,target_names = target_names))
