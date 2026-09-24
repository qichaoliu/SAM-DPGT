"""
SAM-DPGT: SAM-Guided Dual-Prior Graph Transformer for Few-Shot HSI Classification.

Final version: Original SAM order sequential overlay + clean_seg + 2x2 window adjacency.

Usage:
    python run_final.py

Before running:
    1. Download SAM checkpoint (sam_vit_b_01ec64.pth) and place it in the parent directory.
    2. Set DATA_ROOT below to the directory containing your hyperspectral datasets.
"""
import numpy as np, scipy.io as sio, os, sys, json, hashlib, h5py, time
import torch, torch.nn.functional as F
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from skimage.measure import label as conn_label

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DualPrior_GT_Github import DualPriorGT_Github
from utils import get_Samples_GT, GT_To_One_Hot

time.clock = time.perf_counter
device = torch.device("cuda:0")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', 'HyperImage_data'))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')
SAM_CHECKPOINT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'sam_vit_b_01ec64.pth'))
MASK_CACHE = os.path.join(OUTPUT_DIR, 'sam_mask_cache')
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_HOP=1; N_TRAIN=5; N_VAL=5; N_SEEDS=10
NUM_BLOCKS=1; NUM_HEADS=4; BASE_CHANNELS=128
LR=5e-4; WEIGHT_DECAY=1e-2; MAX_EPOCHS=301
EVAL_INTERVAL=50; EARLY_STOP_PATIENCE=20; GRADIENT_CLIP=1.0

datasets = {
    'IP': {'type': 'mat', 'folder': 'indian', 'data_file': 'Indian_pines_corrected.mat',
           'gt_file': 'Indian_pines_gt.mat', 'data_key': 'indian_pines_corrected',
           'gt_key': 'indian_pines_gt', 'n_classes': 16,
           'class_names': ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
                          'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats',
                          'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat',
                          'Woods', 'Buildings-Grass-Trees-Stones', 'Stone-Steel-Towers']},
    'PU': {'type': 'mat', 'folder': 'paviaU', 'data_file': 'PaviaU.mat',
           'gt_file': 'Pavia_University_gt.mat', 'data_key': 'paviaU',
           'gt_key': 'pavia_university_gt', 'n_classes': 9,
           'class_names': ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
                          'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows']},
    'SAL': {'type': 'mat', 'folder': 'Salinas', 'data_file': 'Salinas_corrected.mat',
            'gt_file': 'Salinas_gt.mat', 'data_key': 'salinas_corrected',
            'gt_key': 'salinas_gt', 'n_classes': 16,
            'class_names': ['Broccoli_green_weeds_1', 'Broccoli_green_weeds_2', 'Fallow',
                           'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery',
                           'Grapes_untrained', 'Soil_Vinyard_develop', 'Corn_senesced_green_weeds',
                           'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
                           'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis']},
    'TeaFarm': {'type': 'h5', 'folder': 'TeaFarm', 'data_file': 'Teafarm.mat',
                'gt_file': 'Teafarm_gt.mat', 'data_key': 'data', 'gt_key': 'gt',
                'n_classes': 10,
                'class_names': ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5',
                               'Class_6', 'Class_7', 'Class_8', 'Class_9', 'Class_10']},
}


def get_sam_masks(img_rgb, ds_name):
    """Get SAM masks, using cache if available."""
    hh = hashlib.md5(img_rgb.tobytes()).hexdigest()
    cp = os.path.join(MASK_CACHE, f'sam_masks_{ds_name}_{hh}.npz')
    if os.path.exists(cp):
        d = np.load(cp)
        masks = []
        for i in range(len(d['ious'])):
            masks.append({
                'segmentation': d['segs'][i],
                'predicted_iou': float(d['ious'][i]),
                'stability_score': float(d['stabs'][i]),
                'area': int(d['areas'][i]),
            })
        return masks
    # Run SAM fresh
    os.makedirs(MASK_CACHE, exist_ok=True)
    print(f"    Running SAM for {ds_name}...")
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    try:
        sam.to("cuda:0")
    except:
        pass
    mg = SamAutomaticMaskGenerator(
        model=sam, points_per_side=64, points_per_batch=128,
        pred_iou_thresh=0.80, stability_score_thresh=0.85,
        crop_n_layers=2, crop_n_points_downscale_factor=2,
        min_mask_region_area=50
    )
    try:
        masks = mg.generate(img_rgb)
    except:
        sam.cpu()
        masks = mg.generate(img_rgb)
    segs = np.stack([m['segmentation'] for m in masks])
    ious = np.array([m.get('predicted_iou', 0) for m in masks])
    stabs = np.array([m.get('stability_score', 0) for m in masks])
    areas = np.array([m.get('area', 0) for m in masks])
    np.savez_compressed(cp, segs=segs, ious=ious, stabs=stabs, areas=areas)
    return masks


def partition_original(masks, shape):
    """Original SAM order sequential overlay + clean_seg. No merge, no sorting."""
    ss = np.zeros(shape, dtype=np.int32)
    for i, m in enumerate(masks):
        ss[m['segmentation']] = i + 1
    # clean_seg: keep only largest CC per region
    n0 = int(np.max(ss)) + 1
    clean = np.zeros_like(ss)
    for i in range(n0):
        m = (ss == i)
        lab, nc = conn_label(m, connectivity=1, return_num=True)
        if nc > 1:
            sz = [(c2, np.sum(lab == c2)) for c2 in range(1, nc + 1)]
            clean[lab == max(sz, key=lambda x: x[1])[0]] = i
        else:
            clean[m] = i
    # Relabel consecutively
    unique_labels = np.unique(clean)
    seg = np.zeros_like(clean)
    for new_id, old_id in enumerate(unique_labels):
        seg[clean == old_id] = new_id
    return seg


def build_adjacency(seg, n0):
    """OLD: 2x2 window comparison (more permissive, captures diagonal neighbors)."""
    A = np.zeros([n0, n0], dtype=np.float32)
    for i in range(seg.shape[0] - 1):
        for j in range(seg.shape[1] - 1):
            sub = seg[i:i+2, j:j+2]
            if np.max(sub) != np.min(sub):
                a, b = int(np.max(sub)), int(np.min(sub))
                if a < n0 and b >= 0:
                    A[a, b] = A[b, a] = 1
    A[0, :] = 0; A[:, 0] = 0
    return A


def run_dataset(ds_name, ds_cfg):
    print(f"\n{'='*70}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*70}")

    dp = os.path.join(DATA_ROOT, ds_cfg['folder'], ds_cfg['data_file'])
    gp = os.path.join(DATA_ROOT, ds_cfg['folder'], ds_cfg['gt_file'])
    if ds_cfg['type'] == 'h5':
        with h5py.File(dp, 'r') as f:
            data = np.array(f[ds_cfg['data_key']]).transpose(1, 2, 0)
        with h5py.File(gp, 'r') as f:
            gt = np.array(f[ds_cfg['gt_key']])
    else:
        data = sio.loadmat(dp)[ds_cfg['data_key']]
        gt = sio.loadmat(gp)[ds_cfg['gt_key']]
    if gt.ndim == 3: gt = np.squeeze(gt)
    if gt.ndim == 1: gt = gt.reshape(-1, 1)

    h, w, c = data.shape
    n_classes = ds_cfg['n_classes']
    class_names = ds_cfg['class_names']
    print(f"    Data shape: {data.shape}, Classes: {n_classes}")

    data = preprocessing.StandardScaler().fit_transform(data.reshape(-1, c)).reshape(h, w, c)
    pca = PCA(n_components=3).fit_transform(data.reshape(-1, c)).reshape(h, w, 3)
    rgb = ((pca - pca.min()) / (pca.max() - pca.min()) * 255).astype(np.uint8)

    masks = get_sam_masks(rgb, ds_name)
    print(f"    SAM: {len(masks)} masks")

    seg_final = partition_original(masks, (h, w))
    n0 = int(np.max(seg_final)) + 1

    sizes = np.array([np.sum(seg_final == i) for i in range(n0)])
    print(f"    Partition: {n0} regions, median={np.median(sizes):.0f}px")

    A = build_adjacency(seg_final, n0)
    total_edges = int((A > 0).sum() / 2)
    deg = A.sum(axis=1)
    iso = np.sum(deg == 0)
    print(f"    Graph: {n0} nodes, {total_edges} edges, {iso} isolated ({100*iso/n0:.1f}%)")

    S0 = np.zeros([h * w, n0], dtype=np.float32)
    for x in range(h * w):
        S0[x, seg_final.flat[x]] = 1
    S0t = torch.from_numpy(S0).float()
    ct = torch.clamp(S0t.sum(dim=0), min=1e-15)
    ShT = (S0t / ct).t()
    ST = S0t

    pos = np.zeros((n0, 2), dtype=np.float32)
    for i in range(n0):
        m = (seg_final == i)
        if np.any(m):
            y, x = np.where(m)
            pos[i] = [np.mean(x), np.mean(y)]
    pos_t = torch.from_numpy(pos).float()
    At = torch.from_numpy(A).float()
    ni = torch.from_numpy(data.astype(np.float32)).to(device)

    seed_results = []
    gt_orig = gt.copy()
    for seed in range(N_SEEDS):
        gt = gt_orig.copy()
        print(f"    Seed {seed}: ", end='', flush=True)
        train_gt, test_gt, val_gt = get_Samples_GT(seed, gt, n_classes, N_TRAIN, N_VAL, 'same_num')
        train_oh = torch.from_numpy(np.reshape(GT_To_One_Hot(train_gt, n_classes), [-1, n_classes]).astype(np.float32)).to(device)
        val_oh = torch.from_numpy(np.reshape(GT_To_One_Hot(val_gt, n_classes), [-1, n_classes]).astype(np.float32)).to(device)
        train_mask = torch.from_numpy((train_gt.reshape(-1) != 0).astype(np.float32).reshape(-1, 1)).to(device).repeat(1, n_classes)
        val_mask = torch.from_numpy((val_gt.reshape(-1) != 0).astype(np.float32).reshape(-1, 1)).to(device).repeat(1, n_classes)

        net = DualPriorGT_Github(
            h, w, c, n_classes, n_superpixels=n0, adjacency=At,
            node_positions=pos_t, num_blocks=NUM_BLOCKS, k_hop=K_HOP,
            num_heads=NUM_HEADS, base_channels=BASE_CHANNELS, use_rbf=True
        )
        net.set_assignment_matrices(ShT, ST)
        net.to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best_loss, stop = 99999, 0
        net.train()
        best_state = None
        for epoch in range(MAX_EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            out, _, _ = net(ni)
            out_sm = F.softmax(out, dim=1)
            loss = -torch.sum(torch.mul(train_oh, torch.log(out_sm + 1e-15)) * train_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP)
            optimizer.step()
            if epoch % EVAL_INTERVAL == 0:
                with torch.no_grad():
                    net.eval()
                    out, _, _ = net(ni)
                    out_sm = F.softmax(out, dim=1)
                    val_loss = -torch.sum(torch.mul(val_oh, torch.log(out_sm + 1e-15)) * val_mask)
                    net.train()
                if val_loss < best_loss:
                    best_loss = val_loss
                    stop = 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    stop += 1
                if stop >= EARLY_STOP_PATIENCE:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            out, _, _ = net(ni)
            out_sm = F.softmax(out, dim=1)
            pred = torch.argmax(out_sm, 1).cpu().numpy() + 1
            gt_flat = test_gt.reshape(-1)
            mask = gt_flat != 0
            pred_test = pred[mask]
            gt_test = gt_flat[mask].astype(np.int16)
            oa = np.mean(pred_test == gt_test)
            class_accs = []
            for cls in range(1, n_classes + 1):
                cls_mask = (gt_test == cls)
                if np.sum(cls_mask) > 0:
                    class_accs.append(float(np.mean(pred_test[cls_mask] == cls)))
                else:
                    class_accs.append(0.0)
            aa = np.mean(class_accs)
            kappa = cohen_kappa_score(pred_test, gt_test)
            seed_results.append({
                'seed': seed, 'OA': float(oa), 'AA': float(aa), 'Kappa': float(kappa),
                'class_accs': class_accs
            })
            print(f"OA={oa:.4f} AA={aa:.4f} K={kappa:.4f}")
        del net, best_state
        torch.cuda.empty_cache()

    mean_oa = np.mean([s['OA'] for s in seed_results])
    std_oa = np.std([s['OA'] for s in seed_results])
    mean_aa = np.mean([s['AA'] for s in seed_results])
    std_aa = np.std([s['AA'] for s in seed_results])
    mean_k = np.mean([s['Kappa'] for s in seed_results])
    std_k = np.std([s['Kappa'] for s in seed_results])

    class_means = []
    class_stds = []
    for cls_idx in range(n_classes):
        vals = [s['class_accs'][cls_idx] for s in seed_results]
        class_means.append(np.mean(vals))
        class_stds.append(np.std(vals))

    print(f"\n  >>> {ds_name} Results ({N_SEEDS} seeds):")
    print(f"  OA:    {mean_oa*100:.2f} +/- {std_oa*100:.2f}%")
    print(f"  AA:    {mean_aa*100:.2f} +/- {std_aa*100:.2f}%")
    print(f"  Kappa: {mean_k*100:.2f} +/- {std_k*100:.2f}%")
    print(f"\n  Per-class accuracy:")
    for cls_idx in range(n_classes):
        print(f"    {class_names[cls_idx]:<30s}: {class_means[cls_idx]*100:.2f} +/- {class_stds[cls_idx]*100:.2f}%")

    return {
        'OA': f"{mean_oa*100:.2f} +/- {std_oa*100:.2f}",
        'AA': f"{mean_aa*100:.2f} +/- {std_aa*100:.2f}",
        'Kappa': f"{mean_k*100:.2f} +/- {std_k*100:.2f}",
        'class_accs': {class_names[i]: f"{class_means[i]*100:.2f} +/- {class_stds[i]*100:.2f}" for i in range(n_classes)},
        'n_regions': n0, 'n_edges': total_edges,
        'seed_results': seed_results
    }


if __name__ == '__main__':
    print("=" * 80)
    print("FINAL: Original SAM order + clean_seg + 4-neighbor adjacency")
    print(f"Config: k_hop={K_HOP}, {N_TRAIN} train/class, {N_SEEDS} seeds")
    print("=" * 80)

    all_results = {}
    for ds_name, ds_cfg in datasets.items():
        all_results[ds_name] = run_dataset(ds_name, ds_cfg)
        with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<12s} {'OA (%)':<20s} {'AA (%)':<20s} {'Kappa (%)':<20s} {'Nodes':<8s} {'Edges':<8s}")
    print("-" * 80)
    for ds_name, res in all_results.items():
        print(f"{ds_name:<12s} {res['OA']:<20s} {res['AA']:<20s} {res['Kappa']:<20s} {res['n_regions']:<8d} {res['n_edges']:<8d}")
