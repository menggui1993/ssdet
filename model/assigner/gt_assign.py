import torch
import math

INF = 99999999

def generate_grid_points(h, w, stride):
    """
    Generate anchor points for one scale level
    
    Args:
        h: height of the level
        w: width of the level
        stride: stride of the level
    
    Returns:
        points: shape (w*h, 2)
    """
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    y, x = torch.meshgrid(shifts_y, shifts_x)
    x = torch.reshape(x, [-1])
    y = torch.reshape(y, [-1])
    points = torch.stack([x, y], -1) + stride // 2
    return points

def fcos_assigner(gt_bboxes,
                  gt_labels,
                  points,
                  bg_class_id,
                  strides=(8, 16, 32, 64, 128),
                  level_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
                  center_sample_radius = 1.5
                  ):
    """
    Assign targets to anchor points based on gts and anchors.
    Number of gt from different images are padded to same length N.

    Args:
        gt_bboxes: shape (B, N, 4)
        gt_labels: shape (B, N)
        points: shape [(w1*h1, 2), (w2*h2, 2), ...]
        bg_class_id: index for backbround
        strides: (s1, s2, s3, ...)
    
    Returns:
        cls_targets: shape (B, sum_k(w_k*h_k), 1)
        center_targets: shape (B, sum_k(w_k*h_k), 1)
        reg_targets: shape (B, sum_k(w_k*h_k), 4)
    """
    levels = len(strides)
    batch_size = gt_bboxes.shape[0]
    num_gts = gt_bboxes.shape[1]
    cls_targets = []
    reg_targets = []
    center_targets = []
    for level in range(levels):
        # print(level)
        num_pts = points[level].shape[0]
        x = points[level][:, 0]     #[w*h]
        y = points[level][:, 1]

        l_offset = x[None, :, None] - gt_bboxes[:, :, 0][:, None, :]    #[B,w*h,N]
        t_offset = y[None, :, None] - gt_bboxes[:, :, 1][:, None, :]
        r_offset = gt_bboxes[:, :, 2][:, None, :] - x[None, :, None]
        b_offset = gt_bboxes[:, :, 3][:, None, :] - y[None, :, None]
        offsets = torch.stack((l_offset, t_offset, r_offset, b_offset), dim=-1) #[B,w*h,N,4]
        # print(offsets.shape)
        max_offset = torch.max(offsets, dim=-1)[0]  #[B,w*h,N]
        mask_fit_level = (max_offset > level_ranges[level][0])&(max_offset <= level_ranges[level][1])   #[B,w*h,N]
        
        radius = strides[level] * center_sample_radius
        gt_cx = (gt_bboxes[:, :, 0] + gt_bboxes[:, :, 2]) / 2.0     #[B,N]
        gt_cy = (gt_bboxes[:, :, 1] + gt_bboxes[:, :, 3]) / 2.0     #[B,N]
        gt_w = gt_bboxes[:, :, 2] - gt_bboxes[:, :, 0]              #[B,N]
        gt_h = gt_bboxes[:, :, 3] - gt_bboxes[:, :, 1]              #[B,N]
        x_dis = torch.abs(gt_cx[:, None, :] - x[None, :, None])     #[B,w*h,N]
        y_dis = torch.abs(gt_cy[:, None, :] - y[None, :, None])     #[B,w*h,N]
        gt_w = gt_w.repeat(num_pts, 1, 1).permute(1, 0, 2)          #[B,w*h,N]
        gt_h = gt_h.repeat(num_pts, 1, 1).permute(1, 0, 2)          #[B,w*h,N]
        mask_inside_gt = (x_dis < gt_w / 2.0) & (y_dis < gt_h / 2.0)    #[B,w*h,N]
        mask_inside_center = (x_dis < radius) & (y_dis < radius)        #[B,w*h,N]
        mask_match = mask_inside_gt & mask_inside_center & mask_fit_level   #[B,w*h,N]

        areas = gt_w * gt_h         #[B,w*h,N]
        areas[~mask_match] = INF
        min_area, min_area_ind = torch.min(areas, dim=-1)   #[B,w*h]
        
        cls_target = gt_labels[:,:].gather(1, min_area_ind)   #[B,w*h]
        # print(cls_target.shape)
        cls_target[min_area == INF] = bg_class_id # set to bg             #[B,w*h]
        reg_target = offsets.gather(-2, min_area_ind.reshape(batch_size, num_pts, 1, 1).repeat(1,1,1,4)).squeeze(2) #[B,w*h,4]
        # print(reg_target.shape)
        left_right_max = torch.max(reg_target[:, :, 0], reg_target[:, :, 2])
        left_right_min = torch.min(reg_target[:, :, 0], reg_target[:, :, 2])
        top_bottom_max = torch.max(reg_target[:, :, 1], reg_target[:, :, 3])
        top_bottom_min = torch.min(reg_target[:, :, 1], reg_target[:, :, 3])
        center_target = torch.sqrt((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max))   #[B,w*h]
        # print(center_target.shape)

        cls_targets.append(cls_target)
        reg_targets.append(reg_target)
        center_targets.append(center_target)

    cls_targets = torch.cat(cls_targets, dim=1)
    reg_targets = torch.cat(reg_targets, dim=1)
    center_targets = torch.cat(center_targets, dim=1)
    
    # mask_pos = (cls_targets != bg_class_id)
    # print(torch.count_nonzero(mask_pos, dim=1))

    return cls_targets, center_targets, reg_targets


if __name__ == "__main__":
    input_w, input_h = (800, 800)
    strides = (8, 16, 32, 64, 128)
    points = []
    for stride in strides:
        w = math.ceil(input_w / stride)
        h = math.ceil(input_h / stride)
        points.append(generate_grid_points(h, w, stride))
    gt_bboxes = torch.tensor([[20, 40, 64, 80], 
                              [200, 100, 300, 150],
                              [500, 200, 600, 600]])
    gt_labels = torch.tensor([[3], [2], [1]])

    gt_bboxes = gt_bboxes.repeat(8, 1, 1)
    gt_labels = gt_labels.repeat(8, 1, 1).squeeze(-1) 
    cls_targets, center_targets, reg_targets = fcos_assigner(gt_bboxes, gt_labels, points, 10)
    print(cls_targets.shape)
    print(center_targets.shape)
    print(reg_targets.shape)