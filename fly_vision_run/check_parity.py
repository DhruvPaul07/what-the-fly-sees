import time, numpy as np, flyeye
import flycns.flyvis as fv
from flycns.dynamics.graded import GradedReference

d = np.load('lattice.npz', allow_pickle=True)
lat = {k: d[k] for k in d.files}
ens = fv.load_ensemble()
t = time.time()
net = flyeye.FlyVis(lat, ens, model=0, dt=1/100)
print(f'my network built in {time.time()-t:.1f}s; weights nnz={net.W.nnz}')

# flycns's own constructor on the same lattice (it raises if any edge lacks a trained group)
fl = flyeye.lattice_for_flycns(lat, ens)
gnet = fv.lattice_network(fl, ens, model=0)
ref = GradedReference(gnet, dt_s=1/100)
assert np.allclose(np.sort(gnet.weight), np.sort(net.weight)), 'weights differ'

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, size=(40, 721))
t = time.time(); vr, act_ref = ref.run(x); t_ref = time.time() - t
v = net.rest(1)
mine = np.zeros((40, net.n), np.float32)
t = time.time()
for k in range(40):
    v = net.step(v, x[k][:, None]); mine[k] = v[:, 0]
t_mine = time.time() - t
print(f'max |mine - flycns reference| over 45,669 neurons x 40 steps: {np.abs(mine - act_ref).max():.2e}'
      f'  (activity range {act_ref.min():.2f}..{act_ref.max():.2f})')
print(f'speed: flycns reference {t_ref/40*1000:.0f} ms/step, mine {t_mine/40*1000:.0f} ms/step (batch 1)')
# batched speed
v = net.rest(16); t = time.time()
for k in range(20): v = net.step(v, np.full((721, 16), 0.5, np.float32))
print(f'mine, batch of 16 clips: {(time.time()-t)/20*1000:.0f} ms/step')
