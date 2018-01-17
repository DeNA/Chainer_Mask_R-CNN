import glob
fnames = glob.glob('COCO/train2017/*.jpg')

with open("COCO/train2017.txt", "w") as f:
    for fname in fnames:
        f.write(fname.split('/')[-1].split('.')[0]+'\n')
f.close()

fnames = glob.glob('COCO/val2017/*.jpg')

with open("COCO/val2017.txt", "w") as f:
    for i, fname in enumerate(fnames):
        f.write(fname.split('/')[-1].split('.')[0]+'\n')
        if i > 1000:
            break
f.close()

