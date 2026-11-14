import time, numpy as np, clips
t = time.time()
canv = {n: clips.load_gray(n) for n in clips.TRAIN_IMAGES + clips.TEST_IMAGES}
print(f"loaded {len(canv)} photos in {time.time()-t:.1f}s; sizes", {k: v.shape for k, v in list(canv.items())[:4]})
t = time.time()
train, ktr = clips.make_set({"pan": .40, "glide": .35, "blocks": .15, "grating": .05, "still": .05}, 384,
                            clips.TRAIN_IMAGES, seed=1, canvases=canv)
print(f"train: {train.shape} in {time.time()-t:.0f}s", dict(zip(*np.unique(ktr, return_counts=True))))
t = time.time()
test_nat, kn = clips.make_set({"pan": .5, "glide": .5}, 48, clips.TEST_IMAGES, seed=2, canvases=canv)
test_syn, ks = clips.make_set({"blocks": .5, "grating": .5}, 48, clips.TEST_IMAGES, seed=3, canvases=canv)
test_still, kst = clips.make_set({"still": 1.0}, 8, clips.TEST_IMAGES, seed=4, canvases=canv)
print(f"tests in {time.time()-t:.0f}s")
np.savez_compressed("clips.npz", train=train.astype(np.float16), train_kind=ktr,
                    test_nat=test_nat.astype(np.float16), test_syn=test_syn.astype(np.float16),
                    test_still=test_still.astype(np.float16))
print("eye-input range", float(train.min()), float(train.max()))
