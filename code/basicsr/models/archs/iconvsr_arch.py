import torch
from torch import nn 
import torch.nn.functional as F 
from basicsr.models.archs.spynet_arch import SpyNet
from basicsr.models.archs.basicvsr_arch import ConvResBlock, PSUpsample
from basicsr.models.archs.edvr_arch import PredeblurModule, PCDAlignment, TSAFusion
from basicsr.models.archs.arch_util import ResidualBlockNoBN, flow_warp, make_layer 

class IconVSR(nn.Module):
    """IconVSR network for video super-resolution.
    Args:
        num_feat (int): Channel number of intermediate features. 
            Default: 64.
        num_block (int): Block number of residual blocks in each propagation branch.
            Default: 30.
        keyframe_stride (int): Number determining the keyframes. If stride=5,
            then the (0, 5, 10, 15, ...)-th frame will be the keyframes.
            Default: 5.
        temporal_padding (int): Number of frames to be padded at two ends of
            the sequence. 2 for REDS and 3 for Vimeo-90K. Default: 2
        spynet_path (str): The path of Pre-trained SPyNet model.
            Default: None.
    """
    def __init__(self, 
                      num_feat=64, num_block=30, 
                      keyframe_stride=5, temporal_padding=2, 
                      spynet_path=None):
        super(IconVSR, self).__init__()

        self.num_feat = num_feat
        self.t_pad = temporal_padding
        self.kframe_stride = keyframe_stride

        self.edvr = EDVRExtractor(num_frame=temporal_padding*2 + 1,
                                      center_frame_idx=temporal_padding)
        
        # Flow-based Feature Alignment
        self.spynet = SpyNet(load_path=spynet_path)

        # Coupled Propagation and Information-refill
        self.backward_fuse = nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1, bias=True)
        self.backward_resblocks = ConvResBlock(num_feat + 3, num_feat, num_block)

        self.forward_fuse = nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1, bias=True)
        self.forward_resblocks = ConvResBlock(num_feat * 2 + 3, num_feat, num_block)

        # Pixel-Shuffle Upsampling
        self.up1 = PSUpsample(num_feat, num_feat, scale_factor=2)
        self.up2 = PSUpsample(num_feat, 64, scale_factor=2)

        # The channel of the tail layers is 64
        self.conv_hr = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # Global Residual Learning
        self.img_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # Activation Function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def comp_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Args:
            lrs (tensor): LR frames, the shape is (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 
            forward_flow refers to the flow from current frame to the previous frame. 
            backward_flow is the flow from current frame to the next frame.
        """
        n, t, c, h, w = lrs.size()
        forward_lrs = lrs[:, 1:, :, :, :].reshape(-1, c, h, w) # 'n t c h w -> (n t) c h w'
        backward_lrs = lrs[:, :-1, :, :, :].reshape(-1, c, h, w) # 'n t c h w -> (n t) c h w')
        
        forward_flow = self.spynet(forward_lrs, backward_lrs).view(n, t-1, 2, h, w)
        backward_flow = self.spynet(backward_lrs, forward_lrs).view(n, t-1, 2, h, w)

        return forward_flow, backward_flow

    def extract_refill_features(self, lrs, keyframe_idx):
        """Compute the features for information refill.

        We use EDVR-M to extract features from the selected keyframes
        and its neighbor. The window size in EDVR-M is 5 for REDS and
        7 for Vimeo-90K (following the settings in EDVR).

        Args:
            lrs (Tensor): The input LR sequence with shape (n, t, c, h, w).
            keyframe_idx (list[int]): List of the indices of the selected
                keyframes.

        Returns:
            dict: The features for information-refill. The keys are the
                corresponding index.

        """
        lrs_start = lrs[:, 1+self.t_pad : 1+self.t_pad*2].flip(1)
        lrs_end = lrs[:, -1-self.t_pad*2 : -1-self.t_pad].flip(1)
        lrs = torch.cat([lrs_start, lrs, lrs_end], dim=1)
        num_frame = 2 * self.t_pad + 1

        refill_feat = {}
        for i in keyframe_idx:
            refill_feat[i] = self.edvr(lrs[:, i:i + num_frame].contiguous())
        return refill_feat
    
    def spatial_padding(self, lrs):
        """ Apply spatial pdding.

        Since the PCD module in EDVR requires a resolution of a multiple of 4, 
        we use reflect padding on the LR frame to match the requirements..

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).

        """
        n, t, c, h, w = lrs.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        lrs = lrs.view(-1, c, h, w)
        lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode='reflect')

        return lrs.view(n, t, c, h + pad_h, w + pad_w)
    
    def forward(self, lrs):
        n, t, c, h_in, w_in = lrs.size()
        assert h_in >= 64 and w_in >= 64, (
            'The height and width of input should be at least 64, '
            f'but got {h_in} and {w_in}.')
        
        # Padding
        lrs = self.spatial_padding(lrs)
        h, w = lrs.size(3), lrs.size(4)

        # get the keyframe for information-refill
        keyframe_idx = list(range(0, t, self.kframe_stride))
        if keyframe_idx[-1] != t-1:
            keyframe_idx.append(t-1) # the last frame is a keyframe
        
        # compute flow and refill
        forward_flow, backward_flow = self.comp_flow(lrs)
        refill_feat = self.extract_refill_features(lrs, keyframe_idx)

        # backward propgation
        rlt = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        for i in range(t-1, -1, -1):
            curr_lr = lrs[:, i, :, :, ]
            if i < t-1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, refill_feat[i]], dim=1)
                feat_prop = self.backward_fuse(feat_prop)
            feat_prop = torch.cat([feat_prop, curr_lr], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)
            rlt.append(feat_prop)
        rlt = rlt[::-1]

        # forward propgation
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i-1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, refill_feat[i]], dim=1)
                feat_prop = self.forward_fuse(feat_prop)
            feat_prop = torch.cat([curr_lr, rlt[i], feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # Upsampling
            sr_rlt = self.lrelu(self.up1(sr_rlt))
            sr_rlt = self.lrelu(self.up2(sr_rlt))
            sr_rlt = self.lrelu(self.conv_hr(sr_rlt))
            sr_rlt = self.conv_last(sr_rlt)

            # Global Residual Learning
            base = self.img_up(curr_lr)

            sr_rlt += base
            rlt[i] = sr_rlt
        return torch.stack(rlt, dim=1)[:, :, :, :4 * h_in, :4 * w_in]

class EDVRExtractor(nn.Module):
    """EDVR feature extractor for information-refill in IconVSR.

    We use EDVR-M in IconVSR.

    Paper:
    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: Middle of input frames.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_frame=5,
                deformable_groups=8, num_extract_block=5,
                center_frame_idx=None, hr_in=None, 
                with_predeblur=False, with_tsa=True):
        super(EDVRExtractor, self).__init__()

        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa

        # extract features for each frame
        if self.with_predeblur:
            self.pre_deblur = PredeblurModule(num_feat=num_feat, hr_in=self.hr_in)
            self.conv_1x1 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        
        # extract pyramid features 
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=2, padding=1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=2, padding=1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        
        if self.with_tsa:
            self.fusion = TSAFusion(
                num_feat=num_feat,
                num_frame=num_frame,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x):
        n, t, c, h, w = x.size()

        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, (
                'The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, (
                'The height and width must be multiple of 4.')
        
        # extract features for each frame
        # Level 1
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.pre_deblur(x.view(-1, c, h, w)))
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        
        feat_l1 = self.feature_extraction(feat_l1)

        # Level 2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))

        # Level 3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(n, t, -1, h, w)
        feat_l2 = feat_l2.view(n, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(n, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (n, t, c, h, w)

        if not self.with_tsa:
            aligned_feat = aligned_feat.view(n, -1, h, w)
        feat = self.fusion(aligned_feat)

        return feat

if __name__ == '__main__':
    model = IconVSR()
    lrs = torch.randn(3, 4, 3, 64, 64)
    rlt = model(lrs)
    print(rlt.size())
        

        


