"""CIRKD transforms with an explicit, worker-independent augmentation sequence."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import hashlib
import random

import cv2
import numpy as np
import torch

from ..data import LABEL_IDS

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
MEAN=np.array([104.00698793,116.66876762,122.67891434])


def make_plan(count, *, seed=1, horizon=1280000, crop=512, shape=(1024,2048)):
    """Match serial CIRKD random/NumPy draws without touching model RNG."""
    if not 0<count<=horizon:raise ValueError("Invalid sample budget")
    order=torch.randperm(horizon,generator=torch.Generator().manual_seed(seed))[:count].numpy()
    py=random.Random(seed);np_rng=np.random.RandomState(seed)
    plan=np.empty((count,5),dtype=np.int32)
    for i,position in enumerate(order):
        scale_index=py.randint(0,15);scale=.5+scale_index/10.
        height,width=(max(crop,round(d*scale)) for d in shape)
        plan[i]=(position,scale_index,py.randint(0,height-crop),py.randint(0,width-crop),int(np_rng.choice(2))*2-1)
    return plan


def plan_hash(plan):
    return hashlib.sha256(plan.tobytes()).hexdigest()


class PlannedDataset:
    def __init__(self, root, list_path, plan=None, *, crop=512):
        self.root=root;self.plan=plan;self.crop=crop
        self.rows=[line.split() for line in list_path.read_text().splitlines() if line.strip()]

    def __len__(self):return len(self.rows) if self.plan is None else len(self.plan)

    def __getitem__(self,index):
        record=None if self.plan is None else self.plan[index]
        row=self.rows[index if record is None else int(record[0])%len(self.rows)]
        image=cv2.imread(str(self.root/row[0]),cv2.IMREAD_COLOR)
        raw=cv2.imread(str(self.root/row[1]),cv2.IMREAD_GRAYSCALE)
        if image is None or raw is None:raise ValueError(f"Unreadable input: {row}")
        label=np.full(raw.shape,-1,dtype=np.int32)
        for train_id,label_id in enumerate(LABEL_IDS):label[raw==label_id]=train_id
        if record is not None:
            scale=.5+int(record[1])/10.
            image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
            label=cv2.resize(label,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
        image=np.asarray(image,np.float32)-MEAN
        if record is not None:
            pad_h=max(self.crop-label.shape[0],0);pad_w=max(self.crop-label.shape[1],0)
            if pad_h or pad_w:
                image=cv2.copyMakeBorder(image,0,pad_h,0,pad_w,cv2.BORDER_CONSTANT,value=(0.,0.,0.))
                label=cv2.copyMakeBorder(label,0,pad_h,0,pad_w,cv2.BORDER_CONSTANT,value=-1)
            top,left,flip=map(int,record[2:])
            image=image[top:top+self.crop,left:left+self.crop][:,::flip]
            label=label[top:top+self.crop,left:left+self.crop][:,::flip]
        image=torch.from_numpy(image.transpose(2,0,1).astype(np.float32).copy())
        label=torch.from_numpy(label.astype(np.int64).copy())
        if not torch.isfinite(image).all() or not (label!=-1).any():raise ValueError("Invalid transformed sample")
        return image,label,row[1].rsplit('/',1)[-1].removesuffix('.png')


def batches(dataset, start, end, batch_size, *, workers=4, prefetch=2):
    """Bounded ordered prefetch; only the consumed cursor belongs in checkpoints."""
    if start<0 or end>len(dataset) or (end-start)%batch_size:raise ValueError("Incomplete batch range")
    if workers==0:
        for position in range(start,end,batch_size):
            yield collate([dataset[i] for i in range(position,position+batch_size)])
        return
    queue=deque();next_position=start
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            while queue or next_position<end:
                while len(queue)<prefetch and next_position<end:
                    queue.append([pool.submit(dataset.__getitem__,i) for i in range(next_position,next_position+batch_size)])
                    next_position+=batch_size
                yield collate([future.result() for future in queue.popleft()])
        finally:
            for group in queue:
                for future in group:future.cancel()


def collate(samples):
    return torch.stack([s[0] for s in samples]),torch.stack([s[1] for s in samples]),[s[2] for s in samples]


def calibration_digest_update(digest,images,labels,names):
    for name in names:digest.update(name.encode())
    for tensor in (images,labels):digest.update(tensor.contiguous().numpy().tobytes())


def batch_digest(images,labels,names):
    digest=hashlib.sha256();calibration_digest_update(digest,images,labels,names)
    return digest.hexdigest()
