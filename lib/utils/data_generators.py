from __future__ import absolute_import
from __future__ import division
# import numpy as np
# import cv2
import random
from . import data_augment
from .bbox_transform import *

def bbox_iou(boxA, boxB):
	# single boxA and a set of boxB: x1, y1, x2, y2
	# return A's largest IoU, 1 is excluded.
	boxA = boxA.astype(np.float32)
	boxB = boxB.astype(np.float32)
	wA = boxA[2] - boxA[0] + 1
	wB = boxB[:, 2] - boxB[:, 0] + 1
	hA = boxA[3] - boxA[1] + 1
	hB = boxB[:, 3] - boxB[:, 1] + 1
	SA = wA * hA
	SB = wB * hB
	xmin_max = np.maximum(boxA[0], boxB[:, 0])
	xmax_min = np.minimum(boxA[2], boxB[:, 2])
	ymin_max = np.maximum(boxA[1], boxB[:, 1])
	ymax_min = np.minimum(boxA[3], boxB[:, 3])

	I = np.maximum(xmax_min - xmin_max + 1, 0) * np.maximum(ymax_min - ymin_max + 1, 0)
	U = SA + SB - I
	IoU = I / U
	return IoU

# def ae_tag(boxA, boxB, width):
# 	# return tag_pull and tag_push
# 	cA = np.round((boxA[:2]+boxA[2:]) / 2).astype(np.int)
# 	cA_single = cA[1] * width + cA[0]
# 	neighbor = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int) + cA[np.newaxis, :]
# 	Flag = [True] * 4
# 	neighbor_single = neighbor[:, 1] * width + neighbor[:, 0]
# 	cB = np.round((boxB[:, :2] + boxB[:, 2:]) / 2).astype(np.int)
# 	cB_single = cB[:, 1] * width + cB[:, 0]
# 	for i in range(4):
# 		if neighbor_single[i] in cB_single:
# 			Flag[i] = False
# 	neighbor_single = neighbor_single[Flag]
#
# 	IoU = bbox_iou(boxA, boxB)
# 	IoU[IoU==1] = 0
#
# 	# pull_box = cB_single[np.logical_and(IoU>0.1, IoU<0.5)]
# 	# push_box = cB_single[IoU>0.5]
# 	pull_box = cB_single[IoU>1]
# 	push_box = cB_single[IoU>0.1]
#
# 	pull_box = np.tile(np.concatenate((pull_box, neighbor_single))[:, np.newaxis], [1, 2])
# 	push_box = np.tile(np.concatenate((push_box, np.array([0], dtype=np.int)))[:, np.newaxis], [1, 2])[:-1, :]
#
# 	pull_box[:, 0] = cA_single
# 	push_box[:, 0] = cA_single
#
# 	return pull_box, push_box

def ae_tag(boxA, boxB, width):
	# return tag_pull and tag_push
	cA = ((boxA[:2]+boxA[2:]) / 2).astype(np.int)
	cA_single = cA[1] * width + cA[0]
	neighbor = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int) + cA[np.newaxis, :]
	Flag = [True] * 3
	neighbor_single = neighbor[:, 1] * width + neighbor[:, 0]
	cB = np.round((boxB[:, :2] + boxB[:, 2:]) / 2).astype(np.int)
	cB_single = cB[:, 1] * width + cB[:, 0]
	for i in range(3):
		if neighbor_single[i] in cB_single:
			Flag[i] = False
	neighbor_single = neighbor_single[Flag]

	IoU = bbox_iou(boxA, boxB)
	IoU[IoU==1] = 0

	# pull_box = cB_single[np.logical_and(IoU>0.1, IoU<0.5)]
	# push_box = cB_single[IoU>0.5]
	pull_box = cB_single[IoU>1]
	push_box = cB_single[IoU>0.3]

	pull_box = np.tile(np.concatenate((pull_box, neighbor_single))[:, np.newaxis], [1, 2])
	push_box = np.tile(np.concatenate((push_box, np.array([0], dtype=np.int)))[:, np.newaxis], [1, 2])[:-1, :]

	pull_box[:, 0] = cA_single
	push_box[:, 0] = cA_single

	return pull_box, push_box

def attract_repel(boxA, boxB, width):
	thrs = 0.3
	cA = ((boxA[:2] + boxA[2:]) / 2).astype(np.int)
	neighbor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int) + cA[np.newaxis, :]
	neighbor_single = neighbor[:, 1] * width + neighbor[:, 0]

	cB = ((boxB[:, :2] + boxB[:, 2:]) / 2).astype(np.int)

	IoU = bbox_iou(boxA, boxB)
	IoU[IoU == 1] = 0
	repel = cB[IoU>thrs][:, np.newaxis, :] + np.array([[[0, 0], [0, 1], [1, 0], [1, 1]]], dtype=np.int)
	repel_single = repel[:, :, 1] * width + repel[:, :, 0]
	repel_single = repel_single[:, np.newaxis, :]

	attract_ind = neighbor_single[np.newaxis, :]
	if repel_single.shape[0] == 0:
		repel_ind = np.tile(repel_single, [1, 2, 1])
	else:
		repel_ind = np.concatenate((np.tile(attract_ind, [repel_single.shape[0], 1])[:, np.newaxis, :], repel_single), axis=1)

	if repel_ind.shape[0] == 0:
		pre_off = np.zeros([0, 2], dtype=np.float32)
	else:
		true_center = ((boxB[:, :2] + boxB[:, 2:]) / 2)
		true_center = true_center[IoU > thrs]
		pre_off = true_center - ((boxA[:2] + boxA[2:]) / 2)[np.newaxis, :]
		pre_off = pre_off[:, ::-1]

	return attract_ind, repel_ind, pre_off

def calc_gt_center(C, img_data):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		# with random, not int(kernel / 2)?
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		# if kernel % 2 == 0 & np.random.randint(0, 2) == 0:
		# 	dx = np.exp(-np.square(np.arange(kernel) - int((kernel - 1) / 2)) / s)
		# else:
		# 	dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])

	down = C.down
	scale = 'hw' if C.not_h_only else 'h'
	r = C.radius
	offset = C.offset

	scale_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 2))
	if scale=='hw':
		scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	if offset:
		offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	if C.tl_corner:
		tl_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
		tl_map[:, :, 1] = 1
	if C.br_corner:
		br_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
		br_map[:, :, 1] = 1
	if C.ada_nms or C.AeD_loss:
		density_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 4))
		density_map[:,:,1] = 1

	seman_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs[:, :4]/down
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
			if C.ada_nms:
				density_map[y1:y2, x1:x2, 1] = 0
			if C.tl_corner:
				tl_map[y1:y2, x1:x2, 1] = 0
			if C.br_corner:
				br_map[y1:y2, x1:x2, 1] = 0

	if C.ae_loss or C.AeD_loss:
		max_ = 512
		num_push = 0
		num_pull = 0
		tag_pull = np.zeros(shape=[0, 2], dtype=np.int)
		tag_push = np.zeros(shape=[0, 2], dtype=np.int)
		tag_pull_out = np.zeros(shape=[max_, 2], dtype=np.int)
		tag_push_out = np.zeros(shape=[max_, 2], dtype=np.int)
		mask_pull, mask_push = np.zeros(max_, dtype=np.uint8), np.zeros(max_, dtype=np.uint8)

	if C.IoU_loss:
		IoU_max_ = 512
		num_attract = 0
		num_repel = 0
		attract = np.zeros(shape=[0, 4], dtype=np.int)
		repel = np.zeros(shape=[0, 2, 4], dtype=np.int)
		attract_out = np.zeros(shape=[IoU_max_, 4], dtype=np.int)
		repel_out = np.zeros(shape=[IoU_max_, 2, 4], dtype=np.int)
		mask_attract, mask_repel = np.zeros(IoU_max_, dtype=np.uint8), np.zeros(IoU_max_, dtype=np.uint8)
		pre_off = np.zeros(shape=(0, 2), dtype=np.float32)
		pre_off_out = np.zeros(shape=(IoU_max_, 2), dtype=np.float32)

	if len(gts)>0:
		gts = gts[:, :4]/down
		for ind in range(len(gts)):
			# x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
			# c_x, c_y = int(round((gts[ind, 0] + gts[ind, 2]) / 2)), int(round((gts[ind, 1] + gts[ind, 3]) / 2))
			c_x = max(1, c_x)
			c_x = min(seman_map.shape[1]-2, c_x)
			c_y = max(1, c_y)
			c_y = min(seman_map.shape[0]-2, c_y)
			dx = gaussian(x2-x1)
			dy = gaussian(y2-y1)
			gau_map = np.multiply(dy, np.transpose(dx))
			# if y2 - y1 != gau_map.shape[0] or x2 - x1 != gau_map.shape[1]:
			# 	a = 1
			# if x1 < 0 or y1 < 0 or x2 > seman_map.shape[1] or y2 > seman_map.shape[0]:
			# 	a = 1
			seman_map[y1:y2, x1:x2,0] = np.maximum(seman_map[y1:y2, x1:x2,0], gau_map)
			seman_map[y1:y2, x1:x2,1] = 1
			if C.old_target:
				seman_map[c_y, c_x, 2] = 1
			else:
				seman_map[c_y:c_y + 2, c_x:c_x + 2, 2] = 1

			if C.tl_corner:
				w = x2 - x1
				# h = y2 - y1
				h = x2 - x1

				gau_map2 = np.multiply(dx, np.transpose(dx))

				lmin = max(0, x1 - int(round(w / 2)))
				lmax = min(tl_map.shape[1], x1 - int(round(w / 2)) + w)
				tmin = max(0, y1 - int(round(h / 2)))
				tmax = min(tl_map.shape[0], y1 - int(round(h / 2)) + h)
				l_shift = 0
				t_shift = 0
				if lmin == 0:
					l_shift = int(round(w/2)) - x1
				if tmin == 0:
					t_shift = int(round(h/2)) - y1
				# tl_map[tmin:tmax, lmin:lmax, 0] = seman_map[y1+t_shift:y2, x1+l_shift:x2,0]
				tl_map[tmin:tmax, lmin:lmax, 0] = np.maximum(tl_map[tmin:tmax, lmin:lmax, 0], gau_map2[t_shift:, l_shift:])
				tl_map[tmin:tmax, lmin:lmax, 1] = 1
				tl_map[y1, x1, 2] = 1

			if C.br_corner:
				w = x2 - x1
				# h = y2 - y1
				h = x2 - x1

				gau_map2 = np.multiply(dx, np.transpose(dx))

				lmin = max(0, x2 - int(round(w / 2)))
				lmax = min(br_map.shape[1], x2 - int(round(w / 2)) + w)
				tmin = max(0, y2 - int(round(h / 2)))
				tmax = min(br_map.shape[0], y2 - int(round(h / 2)) + h)
				l_shift = 0
				t_shift = 0
				if lmax == br_map.shape[1]:
					l_shift = (x2 - int(round(w / 2)) + w) - br_map.shape[1]
					# l_shift = int(round(w / 2)) - x1
				if tmax == br_map.shape[0]:
					t_shift = (y2 - int(round(h / 2)) + h) - br_map.shape[0]
					# t_shift = int(round(h / 2)) - y1
				# tl_map[tmin:tmax, lmin:lmax, 0] = seman_map[y1+t_shift:y2, x1+l_shift:x2,0]
				# print(t_shift, l_shift)
				br_map[tmin:tmax, lmin:lmax, 0] = np.maximum(br_map[tmin:tmax, lmin:lmax, 0],
															 gau_map2[:h-t_shift, :w-l_shift])
				br_map[tmin:tmax, lmin:lmax, 1] = 1
				br_map[y1, x1, 2] = 1

			if scale == 'h':
				if C.old_target:
					scale_map[c_y - 2:c_y + 3, c_x - 2:c_x + 3, 0] = np.log(gts[ind, 3] - gts[ind, 1])
					scale_map[c_y - 2:c_y + 3, c_x - 2:c_x + 3, 1] = 1
				else:
					# scale_map[c_y-1:c_y+3, c_x-1:c_x+3, 0] = np.log(gts[ind, 3] - gts[ind, 1])
					scale_map[c_y:c_y+2, c_x:c_x+2, 0] = np.log(gts[ind, 3] - gts[ind, 1])
					# scale_map[c_y-1:c_y+3, c_x-1:c_x+3, 1] = 1
					scale_map[c_y:c_y+2, c_x:c_x+2, 1] = 1
			elif scale=='w':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='hw':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 2] = 1
			if offset:
				if C.old_target:
					offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
					offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
					offset_map[c_y, c_x, 2] = 1
				else:
					temp_y = np.array([[0, 0], [1, 1]], dtype=np.float32)
					temp_x = np.array([[0, 1], [0, 1]], dtype=np.float32)
					offset_map[c_y:c_y + 2, c_x:c_x + 2, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - temp_y
					offset_map[c_y:c_y + 2, c_x:c_x + 2, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - temp_x
					offset_map[c_y:c_y + 2, c_x:c_x + 2, 2] = 1
			if C.ada_nms or C.AeD_loss:
				# calculate the density
				# gts = np.array([[179.75,  81.25, 205.  , 142.5 ],[230.75,  91.  , 246.25, 128.75]])
				ious = bbox_iou(gts[ind], gts)
				ious[ious==1] = 0
				# print(ious)
				density_map[c_y:c_y+2, c_x:c_x+2, 3] = ious.max()
				density_map[c_y:c_y+2, c_x:c_x+2, 2] = 1
				density_map[y1:y2, x1:x2, 0] = np.maximum(density_map[y1:y2, x1:x2, 0], gau_map)
				density_map[y1:y2, x1:x2, 1] = 1
			if C.ae_loss or C.AeD_loss:
				tag_pull_temp, tag_push_temp = ae_tag(gts[ind], gts, int(C.size_train[1]/down))
				tag_pull = np.concatenate((tag_pull, tag_pull_temp))
				tag_push = np.concatenate((tag_push, tag_push_temp))
				num_pull += tag_pull_temp.shape[0]
				num_push += tag_push_temp.shape[0]
			if C.IoU_loss:
				attract_temp, repel_temp, pre_off_temp = attract_repel(gts[ind], gts, int(C.size_train[1]/down))
				attract = np.concatenate((attract, attract_temp))
				repel = np.concatenate((repel, repel_temp))
				pre_off = np.concatenate((pre_off, pre_off_temp))
				num_attract += attract_temp.shape[0]
				num_repel += repel_temp.shape[0]

	outputs = {}
	outputs['hm'] = seman_map
	outputs['wh'] = scale_map
	if offset:
		outputs['reg'] = offset_map
	if C.tl_corner:
		outputs['hmtl'] = tl_map
	if C.br_corner:
		outputs['hmbr'] = br_map
	if C.ada_nms or C.AeD_loss:
		outputs['density'] = density_map
	if C.ae_loss or C.AeD_loss:
		# tag_pull, tag_push: [max_, 2]
		# mask_pull, mask_push: [max_]
		npull = min(max_, num_pull)
		npush = min(max_, num_push)
		mask_pull[:npull] = 1
		mask_push[:npush] = 1
		if npull < max_:
			tag_pull_out[:npull, :] = tag_pull
		else:
			tag_pull_out = tag_pull[np.random.permutation(num_pull)[:max_]]
		if npush < max_:
			tag_push_out[:npush, :] = tag_push
		else:
			tag_push_out = tag_push[np.random.permutation(num_push)[:max_]]
		outputs['tag_pull'] = tag_pull_out
		outputs['tag_push'] = tag_push_out
		outputs['mask_pull'] = mask_pull
		outputs['mask_push'] = mask_push
	if C.IoU_loss:
		nattract = min(IoU_max_, num_attract)
		nrepel = min(IoU_max_, num_repel)
		mask_attract[:nattract] = 1
		mask_repel[:nrepel] = 1
		if nattract < IoU_max_:
			attract_out[:nattract, :] = attract
		else:
			attract_out = attract[np.random.permutation(num_attract)[:IoU_max_]]
		if nrepel < IoU_max_:
			repel_out[:nrepel, :, :] = repel
		else:
			permutation = np.random.permutation(num_repel)
			repel_out = repel[permutation[:IoU_max_]]
			pre_off_out = pre_off[permutation[:IoU_max_]]
		outputs['attract'] = attract_out
		outputs['repel'] = repel_out
		outputs['mask_attract'] = mask_attract
		outputs['mask_repel'] = mask_repel
		outputs['pre_off'] = pre_off_out
	return outputs

def calc_gt_top(C, img_data,r=2):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/4), int(C.size_train[1]/4), 2))
	seman_map = np.zeros((int(C.size_train[0]/4), int(C.size_train[1]/4), 3))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs/4
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
	if len(gts)>0:
		gts = gts/4
		for ind in range(len(gts)):
			x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			w = x2-x1
			c_x = int((gts[ind, 0] + gts[ind, 2]) / 2)

			dx = gaussian(w)
			dy = gaussian(w)
			gau_map = np.multiply(dy, np.transpose(dx))

			ty = np.maximum(0,int(round(y1-w/2)))
			ot = ty-int(round(y1-w/2))
			seman_map[ty:ty+w-ot, x1:x2,0] = np.maximum(seman_map[ty:ty+w-ot, x1:x2,0], gau_map[ot:,:])
			seman_map[ty:ty+w-ot, x1:x2,1] = 1
			seman_map[y1, c_x, 2] = 1

			scale_map[y1-r:y1+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind,3]-gts[ind,1])
			scale_map[y1-r:y1+r+1, c_x-r:c_x+r+1, 1] = 1
	return seman_map,scale_map

def calc_gt_bottom(C, img_data, r=2):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/4), int(C.size_train[1]/4), 2))
	seman_map = np.zeros((int(C.size_train[0]/4), int(C.size_train[1]/4), 3))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs/4
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
	if len(gts)>0:
		gts = gts/4
		for ind in range(len(gts)):
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			y2 = np.minimum(int(C.random_crop[0] / 4) - 1, y2)
			w = x2 - x1
			c_x = int((gts[ind, 0] + gts[ind, 2]) / 2)
			dx = gaussian(w)
			dy = gaussian(w)
			gau_map = np.multiply(dy, np.transpose(dx))

			by = np.minimum(int(C.random_crop[0]/4)-1, int(round(y2+w/2)))
			ob = int(round(y2+w/2))-by
			seman_map[by-w+ob:by, x1:x2, 0] = np.maximum(seman_map[by-w+ob:by, x1:x2, 0], gau_map[:w-ob, :])
			seman_map[by-w+ob:by, x1:x2, 1] = 1
			seman_map[y2, c_x, 2] = 1

			scale_map[y2-r:y2+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind,3]-gts[ind,1])
			scale_map[y2-r:y2+r+1, c_x-r:c_x+r+1, 1] = 1

	return seman_map,scale_map

def get_data(ped_data, C, batchsize = 8):
	current_ped = 0
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		if current_ped>len(ped_data)-batchsize:
			random.shuffle(ped_data)
			current_ped = 0
		for img_data in ped_data[current_ped:current_ped + batchsize]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=True)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_hybrid(ped_data, emp_data, C, batchsize = 8,hyratio=0.5):
	current_ped = 0
	current_emp = 0
	batchsize_ped = int(batchsize * hyratio)
	batchsize_emp = batchsize - batchsize_ped
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		if current_ped>len(ped_data)-batchsize_ped:
			random.shuffle(ped_data)
			current_ped = 0
		if current_emp>len(emp_data)-batchsize_emp:
			random.shuffle(emp_data)
			current_emp = 0
		for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))

			except Exception as e:
				print ('get_batch_gt:',e)
		for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				# x_img[:, :, 0] -= C.img_channel_mean[0]
				# x_img[:, :, 1] -= C.img_channel_mean[1]
				# x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt_emp:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize_ped
		current_emp += batchsize_emp
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_wider(ped_data, C, batchsize = 8):
	current_ped = 0
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		if current_ped>len(ped_data)-batchsize:
			random.shuffle(ped_data)
			current_ped = 0
		for img_data in ped_data[current_ped:current_ped + batchsize]:
			try:
				img_data, x_img = data_augment.augment_wider(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=True)
				else:
					y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_hybrid_stat(ped_data, emp_data, C, batchsize = 8,hyratio=0.5):
	current_ped = 0
	current_emp = 0
	batchsize_ped = int(batchsize * hyratio)
	batchsize_emp = batchsize - batchsize_ped
	while True:
		igs_batch, gts_batch = [], []
		if current_ped>len(ped_data)-batchsize_ped:
			random.shuffle(ped_data)
			current_ped = 0
		if current_emp>len(emp_data)-batchsize_emp:
			random.shuffle(emp_data)
			current_emp = 0
		for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if img_data['ignoreareas'].shape[0] != 0:
					igs_batch.append(img_data['ignoreareas'])
				if img_data['bboxes'].shape[0] != 0:
					gts_batch.append(img_data['bboxes'])
			except Exception as e:
				print ('get_batch_gt:',e)
		for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if img_data['ignoreareas'].shape[0] != 0:
					igs_batch.append(img_data['ignoreareas'])
				if img_data['bboxes'].shape[0] != 0:
					gts_batch.append(img_data['bboxes'])
			except Exception as e:
				print ('get_batch_gt_emp:',e)
		if len(igs_batch) == 0:
			igs_batch = np.zeros([0, 4])
		else:
			igs_batch = np.concatenate(igs_batch, axis=0)
		if len(gts_batch) == 0:
			gts_batch = np.zeros([0, 4])
		else:
			gts_batch = np.concatenate(gts_batch, axis=0)
		current_ped += batchsize_ped
		current_emp += batchsize_emp
		yield np.copy(igs_batch), np.copy(gts_batch)

def get_data_hybrid_volumn(ped_data, emp_data, C, batchsize = 8,hyratio=0.5, nbins=16):
	current_ped = 0
	current_emp = 0
	batchsize_ped = int(batchsize * hyratio)
	batchsize_emp = batchsize - batchsize_ped
	while True:
		x_img_batch, y_seman_batch, y_offset_batch = [], [], []
		if current_ped>len(ped_data)-batchsize_ped:
			random.shuffle(ped_data)
			current_ped = 0
		if current_emp>len(emp_data)-batchsize_emp:
			random.shuffle(emp_data)
			current_emp = 0
		for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_offset = calc_gt_center_volumn(C, img_data, down=C.down, scale=C.scale, offset=C.offset, nbins=nbins)
				else:
					# only for center
					y_seman = calc_gt_center_volumn(C, img_data,down=C.down, scale=C.scale, offset=False, nbins=nbins)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))

			except Exception as e:
				print ('get_batch_gt:',e)
		for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_offset = calc_gt_center_volumn(C, img_data, down=C.down, scale=C.scale, offset=C.offset, nbins=nbins)
				else:
					y_seman = calc_gt_center_volumn(C, img_data,down=C.down, scale=C.scale, offset=False, nbins=nbins)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt_emp:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize_ped
		current_emp += batchsize_emp
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_offset_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch)]

def calc_gt_center_volumn(C, img_data,r=2, down=4,scale='h',offset=True, nbins=16):
	# predefined by the statistics of dataset
	h, l = 336./down, 40./down
	x = np.log(h / l) / nbins
	n = np.arange(nbins + 1)
	y = l * np.power(np.e, n*x)

	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		# with random, not int(kernel / 2)?
		if kernel % 2 == 0 & np.random.randint(0, 2) == 0:
			dx = np.exp(-np.square(np.arange(kernel) - int((kernel - 1) / 2)) / s)
		else:
			dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	if offset:
		offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), nbins, 4))
	seman_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), nbins, 3))
	seman_map[:,:,:,1] = 1
	if len(igs) > 0:
		igs = igs/down
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			# for scale
			s_center = np.log((igs[ind, 3] - igs[ind, 1]) / l) / x
			s_center = int(min(max(0, s_center), nbins-1))
			s_min = max(0, s_center - 1)
			s_max = min(nbins-1, s_center + 1)
			seman_map[y1:y2, x1:x2, s_min:s_max+1, 1] = 0
	if len(gts)>0:
		gts = gts/down
		for ind in range(len(gts)):
			# x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
			# for scale
			s_center = np.log((gts[ind, 3] - gts[ind, 1]) / l) / x
			s_center = int(min(max(0, s_center), nbins - 1))
			s_min = max(0, s_center - 2)
			s_max = min(nbins-1, s_center + 2)

			dx = gaussian(x2-x1)
			dy = gaussian(y2-y1)
			dz = gaussian(5)

			if s_center == 0 or s_center == 1:
				start = 2 - s_center
				dz = dz[start:]
			if s_center == nbins-1 or s_center == nbins-2:
				end = nbins - s_center - 3
				dz = dz[:end]

			gau_map = np.multiply(dy, np.transpose(dx))

			gau_map = gau_map[:, :, np.newaxis] * dz.T[np.newaxis, :]

			seman_map[y1:y2, x1:x2, s_min:s_max+1, 0] = np.maximum(seman_map[y1:y2, x1:x2, s_min:s_max+1, 0], gau_map)
			seman_map[y1:y2, x1:x2, s_min:s_max+1, 1] = 1
			seman_map[c_y, c_x, s_center, 2] = 1

			if offset:
				offset_map[c_y, c_x, s_center, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
				offset_map[c_y, c_x, s_center, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
				offset_map[c_y, c_x, s_center, 2] = np.log((gts[ind, 3] - gts[ind, 1]) / l) / x - s_center - 0.5
				offset_map[c_y, c_x, s_center, 3] = 1

	if offset:
		return seman_map, offset_map
	else:
		return seman_map

def get_data_hybrid_iou(ped_data, emp_data, C, batchsize = 8,hyratio=0.5):
	current_ped = 0
	current_emp = 0
	batchsize_ped = int(batchsize * hyratio)
	batchsize_emp = batchsize - batchsize_ped
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch, y_iou_batch = [], [], [], [], []
		if current_ped>len(ped_data)-batchsize_ped:
			random.shuffle(ped_data)
			current_ped = 0
		if current_emp>len(emp_data)-batchsize_emp:
			random.shuffle(emp_data)
			current_emp = 0
		for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset, y_iou = calc_gt_center_iou(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
					y_iou_batch.append(np.expand_dims(y_iou, axis=0))

			except Exception as e:
				print ('get_batch_gt:',e)
		for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset, y_iou = calc_gt_center_iou(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
					y_iou_batch.append(np.expand_dims(y_iou, axis=0))
			except Exception as e:
				print ('get_batch_gt_emp:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
			y_iou_batch = np.concatenate(y_iou_batch, axis=0)
		current_ped += batchsize_ped
		current_emp += batchsize_emp
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch), np.copy(y_iou_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def calc_gt_center_iou(C, img_data,r=2, down=4,scale='h',offset=True):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		# with random, not int(kernel / 2)?
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		# if kernel % 2 == 0 & np.random.randint(0, 2) == 0:
		# 	dx = np.exp(-np.square(np.arange(kernel) - int((kernel - 1) / 2)) / s)
		# else:
		# 	dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 2))
	iou_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 5))
	if scale=='hw':
		scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	if offset:
		offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	seman_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs/down
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
	if len(gts)>0:
		gts = gts/down
		for ind in range(len(gts)):
			# x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
			dx = gaussian(x2-x1)
			dy = gaussian(y2-y1)
			gau_map = np.multiply(dy, np.transpose(dx))
			seman_map[y1:y2, x1:x2,0] = np.maximum(seman_map[y1:y2, x1:x2,0], gau_map)
			seman_map[y1:y2, x1:x2,1] = 1
			seman_map[c_y, c_x, 2] = 1

			if scale == 'h':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='w':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='hw':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 2] = 1
			if offset:
				offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
				offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
				offset_map[c_y, c_x, 2] = 1

			iou_map[c_y, c_x, 0] = gts[ind, 1] - c_y
			iou_map[c_y, c_x, 1] = gts[ind, 0] - c_x
			iou_map[c_y, c_x, 2] = gts[ind, 3] - c_y
			iou_map[c_y, c_x, 3] = gts[ind, 2] - c_x
			iou_map[c_y, c_x, 4] = 1

	if offset:
		return seman_map,scale_map,offset_map, iou_map
	else:
		return seman_map, scale_map

def get_data_hybrid_iou_refine(ped_data, emp_data, C, batchsize = 8,hyratio=0.5):
	current_ped = 0
	current_emp = 0
	batchsize_ped = int(batchsize * hyratio)
	batchsize_emp = batchsize - batchsize_ped
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch, y_iou_batch = [], [], [], [], []
		if current_ped>len(ped_data)-batchsize_ped:
			random.shuffle(ped_data)
			current_ped = 0
		if current_emp>len(emp_data)-batchsize_emp:
			random.shuffle(emp_data)
			current_emp = 0
		for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset, y_iou = calc_gt_center_iou_refine(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
					y_iou_batch.append(np.expand_dims(y_iou, axis=0))

			except Exception as e:
				print ('get_batch_gt:',e)
		for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset, y_iou = calc_gt_center_iou_refine(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
					y_iou_batch.append(np.expand_dims(y_iou, axis=0))
			except Exception as e:
				print ('get_batch_gt_emp:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
			y_iou_batch = np.concatenate(y_iou_batch, axis=0)
		current_ped += batchsize_ped
		current_emp += batchsize_emp
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch), np.copy(y_iou_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def calc_gt_center_iou_refine(C, img_data,r=2, down=4,scale='h',offset=True):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		# with random, not int(kernel / 2)?
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		# if kernel % 2 == 0 & np.random.randint(0, 2) == 0:
		# 	dx = np.exp(-np.square(np.arange(kernel) - int((kernel - 1) / 2)) / s)
		# else:
		# 	dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 2))
	iou_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 5))
	if scale=='hw':
		scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	if offset:
		offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	seman_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 4))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs/down
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
	if len(gts)>0:
		gts = gts/down
		for ind in range(len(gts)):
			# x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
			dx = gaussian(x2-x1)
			dy = gaussian(y2-y1)
			gau_map = np.multiply(dy, np.transpose(dx))
			seman_map[y1:y2, x1:x2,0] = np.maximum(seman_map[y1:y2, x1:x2,0], gau_map)
			seman_map[y1:y2, x1:x2,1] = 1
			seman_map[c_y, c_x, 2] = 1

			# weighted for different scale
			seman_map[:, :, 3] = 1
			if y2 - y1 < 40:
				seman_map[y1:y2, x1:x2, 3] = 3

			if scale == 'h':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='w':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='hw':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 2] = 1
			if offset:
				offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
				offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
				offset_map[c_y, c_x, 2] = 1

			iou_map[c_y, c_x, 0] = gts[ind, 1] - c_y
			iou_map[c_y, c_x, 1] = gts[ind, 0] - c_x
			iou_map[c_y, c_x, 2] = gts[ind, 3] - c_y
			iou_map[c_y, c_x, 3] = gts[ind, 2] - c_x
			iou_map[c_y, c_x, 4] = 1

	if offset:
		return seman_map,scale_map,offset_map, iou_map
	else:
		return seman_map, scale_map

def get_data_hybrid_volumn_reg(ped_data, emp_data, C, batchsize = 8,hyratio=0.5, nbins=4):
	current_ped = 0
	current_emp = 0
	batchsize_ped = int(batchsize * hyratio)
	batchsize_emp = batchsize - batchsize_ped
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		if current_ped>len(ped_data)-batchsize_ped:
			random.shuffle(ped_data)
			current_ped = 0
		if current_emp>len(emp_data)-batchsize_emp:
			random.shuffle(emp_data)
			current_emp = 0
		for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center_volumn_reg(C, img_data, down=C.down, offset=C.offset, nbins=nbins)
				else:
					# only for center
					y_seman, y_height = calc_gt_center_volumn_reg(C, img_data,down=C.down, offset=False, nbins=nbins)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))

			except Exception as e:
				print ('get_batch_gt:',e)
		for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center_volumn_reg(C, img_data, down=C.down, offset=C.offset)
				else:
					y_seman, y_height = calc_gt_center_volumn_reg(C, img_data,down=C.down, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt_emp:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize_ped
		current_emp += batchsize_emp
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch), np.copy(y_seman_batch), np.copy(y_height_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def calc_gt_center_volumn_reg(C, img_data,r=2, down=4, offset=True, nbins=4):
	# predefined by the statistics of dataset
	h, l = 336./down, 40./down
	x = np.log(h / l) / nbins
	n = np.arange(nbins)
	y = l * np.power(np.e, (n+1)*x)

	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		# with random, not int(kernel / 2)?
		if kernel % 2 == 0 & np.random.randint(0, 2) == 0:
			dx = np.exp(-np.square(np.arange(kernel) - int((kernel - 1) / 2)) / s)
		else:
			dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
	if offset:
		offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), nbins, 3))
	seman_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), nbins, 3))
	seman_map[:,:,:,1] = 1
	if len(igs) > 0:
		igs = igs/down
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			# for scale
			# s_center = np.log((igs[ind, 3] - igs[ind, 1]) / l) / x
			# s_center = int(min(max(0, s_center), nbins-1))
			# s_min = max(0, s_center - 1)
			# s_max = min(nbins-1, s_center + 1)
			# seman_map[y1:y2, x1:x2, s_min:s_max+1, 1] = 0
			seman_map[y1:y2, x1:x2, :, 1] = 0
	if len(gts)>0:
		gts = gts/down
		for ind in range(len(gts)):
			# x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
			# for scale
			s_center = np.log((gts[ind, 3] - gts[ind, 1]) / l) / x
			s_center = int(min(max(0, s_center), nbins - 1))
			# s_min = max(0, s_center - 1)
			# s_max = min(nbins-1, s_center + 1)

			dx = gaussian(x2-x1)
			dy = gaussian(y2-y1)
			# dz = gaussian(3)

			# if s_center == 0:
			# 	dz = dz[1:]
			# if s_center == nbins-1:
			# 	dz = dz[:-1]

			gau_map = np.multiply(dy, np.transpose(dx))

			# gau_map = gau_map[:, :, np.newaxis] * dz.T[np.newaxis, :]

			# seman_map[y1:y2, x1:x2, s_min:s_max+1, 0] = np.maximum(seman_map[y1:y2, x1:x2, s_min:s_max+1, 0], gau_map)
			# seman_map[y1:y2, x1:x2, s_min:s_max+1, 1] = 1
			# seman_map[c_y, c_x, s_center, 2] = 1
			seman_map[y1:y2, x1:x2, s_center, 0] = np.maximum(seman_map[y1:y2, x1:x2, s_center, 0], gau_map)
			# seman_map[y1:y2, x1:x2, :, 1][seman_map[y1:y2, x1:x2, :, 1] == 1] = 0
			seman_map[y1:y2, x1:x2, s_center, 1] = 2
			seman_map[c_y, c_x, s_center, 2] = 1

			scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
			scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1

			if offset:
				offset_map[c_y, c_x, s_center, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
				offset_map[c_y, c_x, s_center, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
				# offset_map[c_y, c_x, s_center, 2] = np.log((gts[ind, 3] - gts[ind, 1]) / l) / x - s_center - 0.5
				# offset_map[c_y, c_x, s_center, 3] = 1
				offset_map[c_y, c_x, s_center, 2] = 1

	seman_map[seman_map==2] = 1
	if offset:
		return seman_map, scale_map, offset_map
	else:
		return seman_map, scale_map

def get_data_hybrid_refine(ped_data, emp_data, C, batchsize = 8,hyratio=0.5):
	current_ped = 0
	current_emp = 0
	batchsize_ped = int(batchsize * hyratio)
	batchsize_emp = batchsize - batchsize_ped
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		if current_ped>len(ped_data)-batchsize_ped:
			random.shuffle(ped_data)
			current_ped = 0
		if current_emp>len(emp_data)-batchsize_emp:
			random.shuffle(emp_data)
			current_emp = 0
		for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))

			except Exception as e:
				print ('get_batch_gt:',e)
		for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt_emp:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize_ped
		current_emp += batchsize_emp
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch), np.copy(y_seman_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_hybrid_bbox_map(ped_data, emp_data, C, batchsize = 8,hyratio=0.5):
	current_ped = 0
	current_emp = 0
	batchsize_ped = int(batchsize * hyratio)
	batchsize_emp = batchsize - batchsize_ped
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch, y_bbox_map_batch = [], [], [], [], []
		if current_ped>len(ped_data)-batchsize_ped:
			random.shuffle(ped_data)
			current_ped = 0
		if current_emp>len(emp_data)-batchsize_emp:
			random.shuffle(emp_data)
			current_emp = 0
		for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset, y_bbox_map = calc_gt_center_bbox_map(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				y_bbox_map_batch.append(np.expand_dims(y_bbox_map, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))

			except Exception as e:
				print ('get_batch_gt:',e)
		for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset, y_bbox_map = calc_gt_center_bbox_map(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				y_bbox_map_batch.append(np.expand_dims(y_bbox_map, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt_emp:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		y_bbox_map_batch = np.concatenate(y_bbox_map_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize_ped
		current_emp += batchsize_emp
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch), np.copy(y_bbox_map_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def calc_gt_center_bbox_map(C, img_data,r=2, down=4,scale='h',offset=True):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		# with random, not int(kernel / 2)?
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		# if kernel % 2 == 0 & np.random.randint(0, 2) == 0:
		# 	dx = np.exp(-np.square(np.arange(kernel) - int((kernel - 1) / 2)) / s)
		# else:
		# 	dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 2))
	if scale=='hw':
		scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	if offset:
		offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	seman_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
	seman_map[:,:,1] = 1
	bbox_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 2))
	bbox_map[:,:,0] = 1
	if len(igs) > 0:
		igs = igs/down
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
			bbox_map[y1:y2, x1:x2,0] = 0
	if len(gts)>0:
		gts = gts/down
		for ind in range(len(gts)):
			# x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
			dx = gaussian(x2-x1)
			dy = gaussian(y2-y1)
			gau_map = np.multiply(dy, np.transpose(dx))
			seman_map[y1:y2, x1:x2,0] = np.maximum(seman_map[y1:y2, x1:x2,0], gau_map)
			seman_map[y1:y2, x1:x2,1] = 1
			seman_map[c_y, c_x, 2] = 1

			bbox_map[y1:y2, x1:x2, 0] = 0
			bbox_map[y1:y2, x1:x2, 1] = 1

			if scale == 'h':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='w':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='hw':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 2] = 1
			if offset:
				offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
				offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
				offset_map[c_y, c_x, 2] = 1

	if offset:
		return seman_map,scale_map,offset_map,bbox_map
	else:
		return seman_map, scale_map

def get_data_hybrid_var_refine(ped_data, emp_data, C, batchsize = 8,hyratio=0.5):
	current_ped = 0
	current_emp = 0
	batchsize_ped = int(batchsize * hyratio)
	batchsize_emp = batchsize - batchsize_ped
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		if current_ped>len(ped_data)-batchsize_ped:
			random.shuffle(ped_data)
			current_ped = 0
		if current_emp>len(emp_data)-batchsize_emp:
			random.shuffle(emp_data)
			current_emp = 0
		for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))

			except Exception as e:
				print ('get_batch_gt:',e)
		for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt_emp:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize_ped
		current_emp += batchsize_emp
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch), np.copy(y_seman_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_hybrid_var(ped_data, emp_data, C, batchsize = 8,hyratio=0.5):
	current_ped = 0
	current_emp = 0
	batchsize_ped = int(batchsize * hyratio)
	batchsize_emp = batchsize - batchsize_ped
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		if current_ped>len(ped_data)-batchsize_ped:
			random.shuffle(ped_data)
			current_ped = 0
		if current_emp>len(emp_data)-batchsize_emp:
			random.shuffle(emp_data)
			current_emp = 0
		for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center_var(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))

			except Exception as e:
				print ('get_batch_gt:',e)
		for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center_var(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt_emp:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize_ped
		current_emp += batchsize_emp
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def calc_gt_center_var(C, img_data,r=2, down=4,scale='h',offset=True):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		# with random, not int(kernel / 2)?
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		# if kernel % 2 == 0 & np.random.randint(0, 2) == 0:
		# 	dx = np.exp(-np.square(np.arange(kernel) - int((kernel - 1) / 2)) / s)
		# else:
		# 	dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 2))
	if scale=='hw':
		scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	if offset:
		offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 8))
		offset_map[:, :, 4:6] = 1
	seman_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs/down
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
			offset_map[y1:y2, x1:x2, 4:6] = 0
	if len(gts)>0:
		gts = gts/down
		for ind in range(len(gts)):
			# x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
			dx = gaussian(x2-x1)
			dy = gaussian(y2-y1)
			gau_map = np.multiply(dy, np.transpose(dx))
			seman_map[y1:y2, x1:x2,0] = np.maximum(seman_map[y1:y2, x1:x2,0], gau_map)
			seman_map[y1:y2, x1:x2,1] = 1
			seman_map[c_y, c_x, 2] = 1

			if scale == 'h':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='w':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='hw':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 2] = 1
			if offset:
				offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
				offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
				offset_map[c_y, c_x, 2:4] = 1
				offset_map[y1:y2, x1:x2, 6] = np.maximum(offset_map[y1:y2, x1:x2, 3], gau_map)
				offset_map[y1:y2, x1:x2, 7] = offset_map[y1:y2, x1:x2, 6].copy()
				offset_map[y1:y2, x1:x2, 4:6] = 1

	if offset:
		return seman_map,scale_map,offset_map
	else:
		return seman_map, scale_map


def get_data_hybrid_fovea(ped_data, emp_data, C, batchsize = 8,hyratio=0.5):
	current_ped = 0
	current_emp = 0
	batchsize_ped = int(batchsize * hyratio)
	batchsize_emp = batchsize - batchsize_ped
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		if current_ped>len(ped_data)-batchsize_ped:
			random.shuffle(ped_data)
			current_ped = 0
		if current_emp>len(emp_data)-batchsize_emp:
			random.shuffle(emp_data)
			current_emp = 0
		for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center_fovea(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))

			except Exception as e:
				print ('get_batch_gt:',e)
		for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center_fovea(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt_emp:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize_ped
		current_emp += batchsize_emp
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def calc_gt_center_fovea(C, img_data,r=2, down=4,scale='h',offset=True):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		# with random, not int(kernel / 2)?
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		# if kernel % 2 == 0 & np.random.randint(0, 2) == 0:
		# 	dx = np.exp(-np.square(np.arange(kernel) - int((kernel - 1) / 2)) / s)
		# else:
		# 	dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 2))
	if scale=='hw':
		scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	if offset:
		offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	seman_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
	seman_map[:,:,1] = 1
	h, w, _ = seman_map.shape
	if len(igs) > 0:
		igs = igs/down
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
	if len(gts)>0:
		gts = gts/down
		for ind in range(len(gts)):
			# x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
			x_min = int(max(c_x - 0.3 * (x2 - x1), 0))
			x_max = int(min(c_x + 0.3 * (x2 - x1), w))
			y_min = int(max(c_y - 0.3 * (y2 - y1), 0))
			y_max = int(min(c_y + 0.3 * (y2 - y1), h))
			dx = gaussian(x2-x1)
			dy = gaussian(y2-y1)
			gau_map = np.multiply(dy, np.transpose(dx))
			seman_map[y1:y2, x1:x2,0] = np.maximum(seman_map[y1:y2, x1:x2,0], gau_map)
			seman_map[y1:y2, x1:x2,1] = 1
			# seman_map[c_y, c_x, 2] = 1
			seman_map[y_min:y_max, x_min:x_max, 2] = 1

			if scale == 'h':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='w':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='hw':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 2] = 1
			if offset:
				offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
				offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
				offset_map[c_y, c_x, 2] = 1

	if offset:
		return seman_map,scale_map,offset_map
	else:
		return seman_map, scale_map
