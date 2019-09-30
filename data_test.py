from core.dataset import Dataset
import numpy as np

if __name__ == '__main__':
    data = Dataset('train')
    it = iter(data)
    batch = next(it)
    batch_image, b_labels, b_boxes, batch_paths = batch
    label_sbbox, label_mbbox, label_lbbox = b_labels
    sbboxes, mbboxes, lbboxes = b_boxes
    #print(batch_image.shape)
    #print(label_sbbox.shape)
    #print(label_mbbox.shape)
    #print(label_lbbox.shape)
    #print(sbboxes.shape)

    print(batch_paths)
    test_img_sb = sbboxes[1]
    test_img_mb = mbboxes[1]
    test_img_lb = lbboxes[1]
    sb_mask = test_img_sb[:,2] != 0
    mb_mask = test_img_mb[:,2] != 0
    lb_mask = test_img_lb[:,2] != 0

    print('small box')
    img_sb = test_img_sb[sb_mask]
    for i in range(img_sb.shape[0]):
    	xind = int(img_sb[i,0] / 8)
    	yind = int(img_sb[i,1] / 8)
    	mask = 1 - np.all(label_sbbox[1, yind, xind, :, :5]==0, axis=-1)
    	mask = mask.astype(bool)
    	print(mask)
    	print(label_sbbox[1, yind, xind, mask, :5])
    	classes = label_sbbox[1, yind, xind, mask, 5:]
    	print(np.argmax(classes))


    print('m box')
    img_mb = test_img_mb[mb_mask]
    for i in range(img_mb.shape[0]):
    	xind = int(img_mb[i,0] / 16)
    	yind = int(img_mb[i,1] / 16)
    	mask = 1 - np.all(label_mbbox[1, yind, xind, :, :5]==0, axis=-1)
    	mask = mask.astype(bool)

    	print(mask)
    	print(label_mbbox[1, yind, xind, mask, :5])
    	classes = label_mbbox[1, yind, xind, mask, 5:]
    	print(np.argmax(classes))


    print('large box')
    img_lb = test_img_lb[lb_mask]
    for i in range(img_lb.shape[0]):
    	xind = int(img_lb[i,0] / 32)
    	yind = int(img_lb[i,1] / 32)
    	mask = 1 - np.all(label_lbbox[1, yind, xind, :, :5]==0, axis=-1)
    	mask = mask.astype(bool)
    	print(mask)
    	print(label_lbbox[1, yind, xind, mask, :5])
    	classes = label_lbbox[1, yind, xind, mask, 5:]
    	print(np.argmax(classes))

