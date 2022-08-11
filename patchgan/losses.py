import torch
from torch import nn

def tversky(y_true, y_pred, beta, batch_mean=True):
    tp = torch.sum(y_true*y_pred, axis=(1,2,3))
    fn = torch.sum((1. - y_pred)*y_true, axis=(1,2,3))
    fp = torch.sum(y_pred*(1. - y_true), axis=(1,2,3))
    #tversky = reduce_mean(tp)/(reduce_mean(tp) + 
    #                           beta*reduce_mean(fn) + 
    #                           (1. - beta)*reduce_mean(fp))
    tversky = tp/\
        (tp + beta*fn + (1. - beta)*fp)
    
    norm = torch.sum(y_true, axis=(1,2,3)) + torch.sum(y_pred, axis=(1,2,3)) - tp
    if batch_mean:
        return torch.mean((1. - tversky)*norm)
    else:
        return (1. - tversky)*norm

def fc_tversky(y_true, y_pred, beta, gamma=0.75, batch_mean=True):
    smooth = 1
    y_true_pos = torch.flatten(y_true)
    y_pred_pos = torch.flatten(y_pred)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1-y_pred_pos))
    false_pos = torch.sum((1-y_true_pos)*y_pred_pos)
    
    answer = (true_pos + smooth)/(true_pos + beta*false_neg + (1-beta)*false_pos + smooth)
    focal_tversky_loss = torch.pow((1- answer), gamma)
    
    return focal_tversky_loss

adversarial_loss = nn.BCELoss() 

def generator_loss(generated_img, target_img):
    gen_loss = fc_tversky(target_img, generated_img, beta = 0.7, gamma=0.75)
    return gen_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss

