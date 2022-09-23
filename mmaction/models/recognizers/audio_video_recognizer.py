# Copyright (c) OpenMMLab. All rights reserved.
from mmaction.models.heads.base import BaseHead
from ..builder import RECOGNIZERS
from .base import BaseRecognizer

from mmcv.runner import auto_fp16

import torch.nn as nn
import torch

# class AVHead(BaseHead):
#     def __init__(self,
#                  num_classes,
#                  in_channels,
#                  loss_cls=dict(type='CrossEntropyLoss'),
#                  spatial_type='avg',
#                  dropout_ratio=0.5,
#                  init_std=0.01,
#                  **kwargs):
#         super().__init__(num_classes, in_channels, loss_cls, **kwargs)

#         self.spatial_type = spatial_type
#         self.dropout_ratio = dropout_ratio
#         self.init_std = init_std
#         if self.dropout_ratio != 0:
#             self.dropout = nn.Dropout(p=self.dropout_ratio)
#         else:
#             self.dropout = None
#         self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

#         if self.spatial_type == 'avg':
#             # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
#             self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
#         else:
#             self.avg_pool = None

#     def init_weights(self):
#         """Initiate the parameters from scratch."""
#         normal_init(self.fc_cls, std=self.init_std)

#     def forward(self, x):
#         """Defines the computation performed at every call.

#         Args:
#             x (torch.Tensor): The input data.

#         Returns:
#             torch.Tensor: The classification scores for input samples.
#         """
#         # [N, in_channels, 4, 7, 7]
#         if self.avg_pool is not None:
#             x = self.avg_pool(x)
#         # [N, in_channels, 1, 1, 1]
#         if self.dropout is not None:
#             x = self.dropout(x)
#         # [N, in_channels, 1, 1, 1]
#         x = x.view(x.shape[0], -1)
#         # [N, in_channels]
#         cls_score = self.fc_cls(x)
#         # [N, num_classes]
#         return cls_score

class VideoAudioRecognizer(nn.Module):
    def __init__(self,model_video: BaseRecognizer,model_audio: BaseRecognizer,freeze=True):
        super().__init__()
        self.model_video = model_video
        self.model_audio = model_audio
        if freeze:
            for param in self.model_video.parameters():
                param.requires_grad = False
            for param in self.model_audio.parameters():
                param.requires_grad = False


        video_features = model_video.cls_head.in_channels
        audio_features = model_audio.cls_head.in_channels
        assert model_video.cls_head.num_classes == model_audio.cls_head.num_classes
        num_classes = model_video.cls_head.num_classes

        # TODO: Change (self.av_head, self.loss) to a single mmaction2 head object, 
        self.av_head = nn.Linear(in_features = video_features + audio_features, out_features = num_classes) 
        self.loss = nn.CrossEntropyLoss()



    def forward(self, imgs, audios, label=None, return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(imgs, audios, label)

        return self.forward_test(imgs,audios)


   

    def forward_train(self, imgs, audios, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        
        losses = dict()
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        audios = audios.reshape((-1, ) + audios.shape[2:])

        video_features = self.model_video.extract_feat(imgs)
        audio_features = self.model_audio.extract_feat(audios)

        # TODO: Don't just flatten or avg pool, think about the general way to deal with this, 
        # TODO: use mmaction2 Head to handle final features (pre cat).  
        # video_features = video_features.reshape(video_features.size(0),-1)
        video_features = self.model_video.cls_head.avg_pool(video_features)
        video_features = video_features.view(video_features.size(0), -1)
        audio_features = self.model_audio.cls_head.avg_pool(audio_features)
        audio_features = audio_features.view(audio_features.size(0), -1)

        cat_features = torch.cat([video_features, audio_features], dim=1)
        
        cls_score = self.av_head(cat_features)

        # TODO: don't forget labels.squeeze() once you use mmactio2.head
        gt_labels = labels.squeeze(-1)
        
        loss = {"loss_cls": self.loss(cls_score,gt_labels)}
        losses.update(loss)

        return loss

    def forward_test(self, imgs, audios):
        """Defines the computation performed at every call when evaluation and
        testing."""

        # MK TODO: cat features and pass to head.  
        # sol_1: cat features and pass to head, or,
        # sol_2: keep definition the same and change self.extract_feat
        # sol_3: change backbone to do what you want. 

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        audios = audios.reshape((-1, ) + audios.shape[2:])

        video_features = self.model_video.extract_feat(imgs)
        audio_features = self.model_audio.extract_feat(audios)

        # TODO: Don't just flatten or avg pool, think about the general way to deal with this, 
        # TODO: use mmaction2 Head to handle final features (pre cat).  
        # video_features = video_features.reshape(video_features.size(0),-1)
        video_features = self.model_video.cls_head.avg_pool(video_features)
        video_features = video_features.view(video_features.size(0), -1)
        audio_features = self.model_audio.cls_head.avg_pool(audio_features)
        audio_features = audio_features.view(audio_features.size(0), -1)

        cat_features = torch.cat([video_features, audio_features], dim=1)
        
        cls_score = self.av_head(cat_features)

        return cls_score.cpu().numpy()

    def forward_gradcam(self, audios):
        raise NotImplementedError

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        imgs = data_batch['imgs']
        audios = data_batch['audios']
        label = data_batch['label']


        losses = self(imgs, audios, label, return_loss=True)

        loss, log_vars = self.model_video._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        imgs = data_batch['imgs']
        audios = data_batch['audios']
        label = data_batch['label']


        losses = self(imgs, audios, label, return_loss=True)

        loss, log_vars = self.model_video._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
