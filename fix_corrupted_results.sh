#!/bin/bash
# fix_corrupted_results.sh — repair post-reboot corruption:
#   (1) Delete checkpoint dirs whose pytorch_model.bin is truncated/empty.
#       Trainer will then resume from an earlier good checkpoint or restart.
#   (2) Repair all_results.json files where the post-reboot final eval
#       overwrote the best_metric with a worse value.
#
# IDEMPOTENT: safe to run repeatedly. Only touches corrupted/incorrect data.

set -e
cd "$(dirname "$0")"

# SAFETY: refuse to run while training is actively writing checkpoints,
# since a half-written pytorch_model.bin would be misclassified as corrupt
# and deleted. Only safe to run during reboot recovery (training dead) or
# when explicitly forced after manually stopping training.
if pgrep -f "run_glue.py" >/dev/null && [ "${1:-}" != "--force" ]; then
    echo "REFUSING: a run_glue.py process is alive."
    echo "  Either wait for it to finish, kill it, or pass --force to override."
    exit 1
fi

echo "Step 1: scanning for corrupted checkpoints..."
.venv/bin/python << 'EOF'
import os, torch
removed = 0
for root, dirs, files in os.walk('NLU/output/glue'):
    if 'pytorch_model.bin' not in files:
        continue
    if 'checkpoint-' not in os.path.basename(root):
        continue
    f = os.path.join(root, 'pytorch_model.bin')
    if os.path.getsize(f) == 0:
        # zero-byte: definitely corrupt
        import shutil; shutil.rmtree(root)
        print(f"  DELETED zero-byte ckpt: {root}")
        removed += 1
        continue
    try:
        torch.load(f, map_location='cpu', weights_only=True)
    except Exception as e:
        import shutil; shutil.rmtree(root)
        print(f"  DELETED corrupt ckpt ({type(e).__name__}): {root}")
        removed += 1
print(f"Step 1 removed {removed} corrupted checkpoints")
EOF

echo ""
echo "Step 2: repairing all_results.json files..."
.venv/bin/python << 'EOF'
import json, os, tempfile

def atomic_write_json(path, data):
    """Write JSON atomically: write to a sibling temp file, fsync, rename.
    A power failure mid-write leaves either the old content or the new — never partial."""
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix='.tmp_aw_', dir=d)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise

fixed = 0
for d in sorted(os.listdir('NLU/output/glue')):
    if not d.startswith(('lora_', 'map_', 'hp_')):
        continue
    ar = f'NLU/output/glue/{d}/model/all_results.json'
    ts = f'NLU/output/glue/{d}/model/trainer_state.json'
    if not (os.path.exists(ar) and os.path.exists(ts)):
        continue
    try:
        r = json.load(open(ar))
        s = json.load(open(ts))
    except Exception as e:
        print(f"  SKIP {d}: {e}")
        continue
    m_keys = [k for k in r if k.startswith('eval_') and all(x not in k for x in ['loss','runtime','sample','mem'])]
    if not m_keys:
        continue
    m_key = m_keys[0]
    reported = r[m_key]
    best = s.get('best_metric', 0) or 0
    if best > reported + 1e-4:
        bm_step = None
        if s.get('best_model_checkpoint'):
            try: bm_step = int(s['best_model_checkpoint'].split('-')[-1])
            except Exception: pass
        be = next((e for e in s['log_history']
                   if e.get('step') == bm_step and m_key in e), None) if bm_step else None
        if be:
            r[m_key] = be[m_key]
            r['eval_loss'] = be.get('eval_loss', r.get('eval_loss'))
            r['epoch'] = be.get('epoch', r.get('epoch'))
        else:
            r[m_key] = best
        r['_note'] = 'corrected from trainer_state best_metric (post-reboot recovery)'
        atomic_write_json(ar, r)
        print(f"  FIXED {d}: {reported*100:.2f} -> {r[m_key]*100:.2f}")
        fixed += 1
print(f"Step 2 fixed {fixed} all_results.json files")
EOF
