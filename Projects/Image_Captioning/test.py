import os
checkpoint = None
traincheckpoint = None
if os.path.exists('/media/fadhla/Ubunt_2/caption_results/checkpoint/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'):
    checkpoint = '/media/fadhla/Ubunt_2/caption_results/checkpoint/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'

if os.path.exists('/media/fadhla/Ubunt_2/caption_results/checkpoint/TRAIN_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'):
    traincheckpoint = '/media/fadhla/Ubunt_2/caption_results/checkpoint/TRAIN_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'

print(checkpoint)
print(traincheckpoint)