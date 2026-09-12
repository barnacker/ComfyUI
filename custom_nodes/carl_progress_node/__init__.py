"""carl_progress v2 — full pipeline telemetry for ComfyUI renders.

v1 (09-09): sampler step telemetry + GET /carl/progress.
v2 (09-09, user: 'track more granular than by steps... VAEDecode, CreateVideo,
SaveVideo didn't become green and we don't know what percent they were'):

  1. SUB-STEP units — transformer block forwards inside each sampler step
     (WanAttentionBlock.forward = the live unit: every denoising step is
     blocks_total_forwards / steps passes over the full token sequence).
  2. VAE DECODE % — WanVAEDecoder3d.forward = one temporal chunk (the Wan VAE
     decodes latent-t in 1 + t//2 chunks; each forward call IS a counted chunk).
  3. ENCODE phase + mp4 bytes-on-disk (CreateVideo/SaveVideo run as a phase
     with measured wall + output growth; no io-module internal hooks).
  4. Phases drive the dashboard graph: VAEDecode / CreateVideo / SaveVideo
     get real active/done states from /carl/progress, not just the probe.

Same survivability contract as v1: custom_nodes/ only (git-unmanaged —
survives git pull/rebuild; rollback = delete folder + restart). Zero core
file edits: every hook wraps-and-delegates (originals always run).
State: one overwrite file (no growing logs) — real path is
<C:\carl-video>\ComfyUI\carl_progress.json (dirname x3 of this file).
Kill switch: env CARL_PROGRESS=0. Boot seeds idle.

Route payload:  {"state": idle|sampling|decode|encode|done,
                 "detail": {ts, jobs[], history[],
                            decode:{state, chunks_done, chunks_total, pct, s_per_chunk, eta_s, elapsed},
                            encode:{state, elapsed, clip_bytes, clip_file}}}
"""
import glob
import json
import logging
import os
import threading
import time

try:
    from aiohttp import web
    _WEB_OK = True
except Exception:
    web = None
    _WEB_OK = False

# custom_nodes/carl_progress_node/__init__.py → 3× dirname = the ComfyUI dir
_COMFYUI_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STACK_ROOT = os.path.dirname(_COMFYUI_ROOT)   # C:\carl-video
STATE_PATH = os.path.join(_COMFYUI_ROOT, "carl_progress.json")   # v1-verified live path
_OUT_CANDIDATES = [os.path.join(_STACK_ROOT, "output", "carl"), os.path.join(_STACK_ROOT, "output")]
ENABLED = os.environ.get("CARL_PROGRESS", "1") != "0"

# Hook-only module: no UI node classes. The empty-mapping marker is what
# ComfyUI's loader requires to count this module as loaded (without it the
# loader logs a spurious "IMPORT FAILED" even though the hooks install).
NODE_CLASS_MAPPINGS = {}
WEB_DIRECTORY = None

_lock = threading.RLock()
# one global timeline object — a render is one job chain (sampler stages ->
# decode -> encode); jobs[] = sampler stages, the rest = chain phases.
_live = {
    "jobs": [],
    "history": [],
    "decode": None,   # {"state", "chunks_done", "chunks_total", "chunk_t0", "chunk_last", "elapsed..."}
    "encode": None,   # {"state", "started", "mp4_tmark", "clip_bytes", "clip_file"}
}
_prev_mp4_ts = None   # armed marker: newest pre-existing mp4 mtime
_idle = [0]           # monitor: consecutive fully-settled ticks
_enc_idle = [0]       # monitor: consecutive non-growing ticks for current clip


_WS_CLIENTS = set()   # live aiohttp websockets for the /carl/progress/ws push
_LOOP = None          # ComfyUI asyncio loop, captured at route registration

# v4 (09-11, user contract: every node counts — single-shot nodes are 1/1
# units "true/false completed", multi-step nodes their smallest unit, global
# % = SUM over all units):
_cur_prompt = [None, 0]   # [prompt_id, node_count] — stamped at PromptQueue.put
_exec_prompt = [None]     # prompt_id whose nodes are being counted now
_nodes_exec = set()       # completed display-node ids of the active prompt
_nodes_last = set()       # last closed prompt's completed ids (idle bar)
_STALE_EMPTY = frozenset()

# v4.1: per-hook self-diagnostics, surfaced in the state file payload so a
# hook that fails to install (or never fires) is visible in the dashboard
# instead of a silently-zero counter. Each entry:
#   {import: bool, installed: bool, calls: int, first_err: str|None}
# Incrementing `calls` inside the wrap is the "the hook actually fires"
# proof — import+installed can be true while calls stays 0 if the runtime
# class is never the one being called (the LTX AV-block case).
_diag = {}


_diag_lock = threading.Lock()   # guards _diag, independent of the state _lock


def _diag_note(name, **kw):
    # `import` is a reserved word, so callers pass imp=True/False and it is
    # mapped onto the documented "import" field here (never stored as "imp").
    if "imp" in kw:
        kw["import"] = kw.pop("imp")
    with _diag_lock:
        d = _diag.setdefault(name, {"import": False, "installed": False,
                                    "calls": 0, "first_err": None})
        d.update(kw)


def _diag_calls(name):
    # Worker-thread path: dedicated lock, no coupling to the state lock.
    try:
        with _diag_lock:
            _diag[name]["calls"] += 1
    except Exception:
        pass


def _diag_snapshot():
    # Read-side view; called from _emit/_live_detail while holding the state
    # _lock, but takes _diag_lock for the copy so it cannot race a writer.
    try:
        with _diag_lock:
            return {k: dict(v) for k, v in _diag.items()}
    except Exception:
        return {}


def _cur_nodes_ids():
    """Completed node ids for the count: the active prompt's accumulating set.
    New prompt not yet firing = empty (its units haven't happened); between
    renders the last closed prompt's frozen set reads the completed bar."""
    if _cur_prompt[0] is None:
        return _nodes_last
    if _exec_prompt[0] == _cur_prompt[0]:
        return _nodes_exec
    return _STALE_EMPTY  # prompt queued/starting: no nodes done YET for it


def _nodes_done_n():
    try:
        return len(_cur_nodes_ids())
    except Exception:
        return 0


def _state_of(detail):
    state = "idle"
    if any(j.get("state") != "done" for j in detail.get("jobs", [])):
        state = "sampling"
    elif (detail.get("decode") or {}).get("state") == "running":
        state = "decode"
    elif (detail.get("encode") or {}).get("state") == "running":
        state = "encode"
    return state


def _ws_push(detail=None):
    """Real-time push from the hook/monitor threads to connected dashboard
    legs. NON-BLOCKING by design: the caller is a GPU thread inside a step,
    so sends are fire-and-forget on the aiohttp loop (run_coroutine_threadsafe)
    — no fut.result() anywhere. Dead sockets drop via done-callback."""
    global _LOOP
    if not _WS_CLIENTS or _LOOP is None or _LOOP.is_closed():
        return
    try:
        import asyncio, inspect

        if detail is None:
            detail = _live_detail()
        msg = json.dumps({"state": _state_of(detail), "detail": detail})

        for ws in list(_WS_CLIENTS):
            try:
                async def _one(ws=ws):
                    res = ws.send_str(msg)
                    if inspect.iscoroutine(res):
                        await res

                def _cb(f, w=ws):
                    try:
                        if f.cancelled() is False:
                            f.exception()      # retrieve & mark-seen; no drop needed
                    except Exception:
                        _WS_CLIENTS.discard(w)

                fut = asyncio.run_coroutine_threadsafe(_one(), _LOOP)
                fut.add_done_callback(_cb)
            except Exception:
                _WS_CLIENTS.discard(ws)
    except Exception:
        pass


def _emit():
    """Atomic overwrite of the state file + WS push. Never raises."""
    with _lock:
        jobs = []
        for j in _live["jobs"]:
            jj = dict(j)
            jj.pop("_last_t", None)
            jj.pop("_be", None)
            jj.pop("_step_now", None)
            jj.pop("_hc", None)
            jj.pop("_hs", None)
            jj.pop("_hn", None)
            jobs.append(jj)
        payload = {
            "ts": time.time(),
            "jobs": sorted(jobs, key=lambda j: j.get("started", 0)),
            "history": list(_live["history"]),
            "decode": dict(_live["decode"]) if _live["decode"] else None,
            "encode": dict(_live["encode"]) if _live["encode"] else None,
            # v4 per-node units: total = graph node count at prompt-put time;
            # done = engine-truth completed display-node ids (send_sync hook).
            "nodes": {"total": int(_cur_prompt[1]), "done": _nodes_done_n(),
                      "ids": sorted(_cur_nodes_ids()) if _cur_prompt[0] is not None else [],
                      "prompt": _cur_prompt[0]},
            # v4.1: per-hook install/fire diagnostics (snapshot; takes _diag_lock).
            "diag": _diag_snapshot(),
        }
    tmp = STATE_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, STATE_PATH)
    except Exception:
        pass
    _ws_push(payload)


def _open_jobs():
    with _lock:
        return [j for j in _live["jobs"] if j.get("state") != "done"]


def _last_open():
    o = _open_jobs()
    return o[-1] if o else None


def _new_job(key, steps, start_step, last_step, blocks_total):
    with _lock:
        prev = next((j for j in _live["jobs"] if j.get("key") == str(list(key)) and j.get("state") == "done"), None)
        seq = (prev.get("seq", 0) if prev else 0) + 1
        if prev is not None:
            _live["history"].append(prev)
            if len(_live["history"]) > 4:
                _live["history"].pop(0)
        _live["jobs"] = [j for j in _live["jobs"] if j is not prev]
        _live["jobs"].append({
            "key": str(list(key)),
            "seq": seq,
            "stage": "%s..%s" % (start_step, last_step if last_step is not None else start_step + steps),
            "steps": int(steps),
            "state": "step0",
            "step_done": -1,
            "started": time.time(),
            "total_steps": int(steps),
            "step_times": [],
            "s_per_step": None,
            "eta_s": None,
            "blocks_total": int(blocks_total or 0),
            "blocks_done": 0,
            "_step_now": {},
        })
        global _prev_mp4_ts
        _prev_mp4_ts = _max_mp4_mtime()
    _emit()


def _max_mp4_mtime():
    try:
        best = 0.0
        for d in _OUT_CANDIDATES:
            for f in glob.glob(os.path.join(d, "*.mp4")):
                try:
                    best = max(best, os.path.getmtime(f))
                except Exception:
                    pass
        return best
    except Exception:
        return 0.0


def _find_new_mp4(tmark):
    try:
        best = (None, 0.0)
        for d in _OUT_CANDIDATES:
            for f in glob.glob(os.path.join(d, "*.mp4")):
                try:
                    mt = os.path.getmtime(f)
                    if mt > tmark and mt > best[1]:
                        best = (f, mt)
                except Exception:
                    pass
        f, mt = best
        if f is None:
            return None
        return {"file": os.path.basename(f), "bytes": os.path.getsize(f), "mt": mt}
    except Exception:
        return None


def _finish(key):
    with _lock:
        for j in _live["jobs"]:
            if j.get("key") == str(list(key)) and j.get("state") != "done":
                j["state"] = "done"
                j["finished"] = time.time()
                j["eta_s"] = None
                j.pop("_last_t", None)
                j.pop("_step_now", None)
                break
    _emit()


def patch():
    """Install hooks. Called once at import time by ComfyUI's node loader."""
    if not ENABLED:
        return
    logging.info("carl_progress v2: installing")
    _emit()  # boot seed: file exists with idle truth from first boot

    # ---- hook 1: sampler stage boundary (common choke point) --------------
    try:
        import nodes
        orig_common = nodes.common_ksampler

        def patched_common(model, seed, steps, cfg, sampler_name, scheduler,
                           positive, negative, latent, denoise=1.0, disable_noise=False,
                           start_step=None, last_step=None, force_full_denoise=False):
            key = (id(model), start_step if start_step is not None else 0,
                   last_step if last_step is not None else -1)
            # blocks per forward pass — chain-walk the wrapper (ModelPatcher ->
            # module -> ...): v2 read getattr(model,'model').blocks and landed
            # on a wrapper without .blocks => blocks_total 0 (cosmesis bug).
            # Engine truth: len(self.blocks) (comfy/ldm/wan/model.py:606).
            nb = 0
            try:
                m = model
                for _ in range(4):
                    if m is None:
                        break
                    try:
                        for attr in ("blocks", "transformer_blocks", "double_blocks"):
                            b = getattr(m, attr, None)
                            if b is not None and len(b):
                                nb = len(b)
                                break
                        if nb:
                            break
                    except Exception:
                        pass
                    nxt = getattr(m, "model", None)
                    if nxt is None or nxt is m:
                        break
                    m = nxt
            except Exception:
                nb = 0
            try:
                _new_job(key, steps, start_step if start_step is not None else 0, last_step, nb * steps)
            except Exception:
                pass
            try:
                return orig_common(model, seed, steps, cfg, sampler_name, scheduler,
                                   positive, negative, latent, denoise=denoise,
                                   disable_noise=disable_noise, start_step=start_step,
                                   last_step=last_step, force_full_denoise=force_full_denoise)
            finally:
                try:
                    _finish(key)
                except Exception:
                    pass
        nodes.common_ksampler = patched_common
        logging.info("carl_progress v2: sampler hook")
    except Exception as e:
        logging.error("carl_progress v2: sampler hook failed: %r", e)

    # ---- hook 2: per-step (preview callback fires after each complete step)
    try:
        import latent_preview
        orig_prepare = latent_preview.prepare_callback

        def patched_prepare(model, steps):
            base_cb = orig_prepare(model, steps)
            mid = id(model)

            def hook(step, x0, x, total_steps):
                try:
                    job = None
                    with _lock:
                        for j in _live["jobs"]:
                            if str(mid) in (j.get("key") or "") and j.get("state") != "done":
                                job = j
                    if job is not None:
                        now = time.time()
                        job["step_done"] = int(step)
                        job["state"] = "sampling"
                        # v3.2 instrumentation (09-10): A0Y ran v3.1 and ended
                        # with a ONE-entry step_times (338.3 = the whole
                        # sampling) while step_done walked to 19 — ~18 hook
                        # calls vanished (model-id matches lost between steps?).
                        # Count every call + last step so the next render
                        # tells us which of {hook not called, job dict
                        # replaced, id mismatch} actually happened.
                        job["_hc"] = int(job.get("_hc", 0)) + 1
                        job["_hs"] = int(step)
                        job["_hn"] = len(job.get("_step_now", {}))
                        # v3.1: ANCHOR-BASED step timing. v3's `_last_t` delta was
                        # losing its baseline on the box (all entries read as
                        # now - started => s_per_step inflated ~10x, feeding the
                        # 71%-vs-35% job% bug, 09-10 A0X). Per-step now-anchor map:
                        # step time = anchor(k) - anchor(k-1), first = anchor(0) -
                        # started. Dedup'd per step, order-independent, no
                        # transient state to lose.
                        try:
                            nows = job.setdefault("_step_now", {})
                            if int(step) not in nows:
                                nows[int(step)] = now
                                keys = sorted(nows)
                                st = []
                                prev = job["started"]
                                for k2 in keys:
                                    st.append(max(0.0, round(nows[k2] - prev, 3)))
                                    prev = nows[k2]
                                job["step_times"] = st
                        except Exception:
                            pass
                        # v3: empirical block denominator. Step 0 completes
                        # with exactly (blocks/layer * cfg-passes) forwards —
                        # the engine only runs forward CHAINS inside sampling
                        # (VAE is a conv net, no attention blocks), so the
                        # step-0 delta is the per-step unit. v2's model
                        # introspection missed the real stack (WAN22 wrapper
                        # has no public .blocks; weights offloaded) — this
                        # counts what actually ran.
                        bd = int(job.get("blocks_done", 0))
                        if int(step) == 0 and bd > 0:
                            # fresh calibration for this stage chain: keep any
                            # previous stage's cumulative in _bc (2-stage 14B),
                            # reset the running count, set the denominator
                            job["_bc"] = int(job.get("_cb", 0))   # carry prior
                            job["_cb"] = bd                        # this stage's step-0 unit
                            job["blocks_done"] = 0
                            job["blocks_total"] = (int(job.get("_bc", 0)) + bd) * int(total_steps)
                        else:
                            job["blocks_done"] = bd
                        ts = job["step_times"]
                        core = ts[1:] if len(ts) > 1 else []
                        base = core if core else ts
                        if base:
                            job["s_per_step"] = round(sum(base) / len(base), 2)
                            remaining = int(total_steps) - int(step) - 1
                            job["eta_s"] = int(remaining * job["s_per_step"]) if remaining > 0 else 0
                        _emit()
                except Exception:
                    pass
                try:
                    return base_cb(step, x0, x, total_steps)
                except Exception:
                    return None
            return hook
        latent_preview.prepare_callback = patched_prepare
        logging.info("carl_progress v2: step hook")
    except Exception as e:
        logging.error("carl_progress v2: step hook failed: %r", e)

    # ---- hook 3: transformer block forwards (sub-step units) --------------
    try:
        import comfy.ldm.wan.model as wm
        orig_fwd = wm.WanAttentionBlock.forward

        def block_fwd(self, *a, **k):
            try:
                j = _last_open()
                if j is not None:
                    j["blocks_done"] = int(j.get("blocks_done", 0)) + 1
                    # engine's own truth (wan/model.py:606) arrives in every
                    # call: layer count, for the label
                    if not j.get("blocks_layers"):
                        to = k.get("transformer_options") or {}
                        lbs = to.get("total_blocks")
                        if lbs:
                            j["blocks_layers"] = int(lbs)
                    # v3.4 emit (09-10 user spec): EMIT at every single block.
                    # The in-memory count is exact per block; the emit is
                    # debounce-gated by DISPLAY REFRESH RATE (33 ms = 30 fps),
                    # never by a block batch: at any block rate the UI sees
                    # every block, and a back-to-back burst (multi-stage
                    # chains, faster hardware) is clamped to one emit per
                    # refresh window instead of 100 writes per frame. No
                    # stall fallback needed: no block => no emit, and the
                    # step-boundary hook always emits the step's end state.
                    now = time.time()
                    if now - j.get("_be", 0.0) >= 0.033:
                        j["_be"] = now
                        _emit()
            except Exception:
                pass
            return orig_fwd(self, *a, **k)
        wm.WanAttentionBlock.forward = block_fwd
        logging.info("carl_progress v2: block hook")
    except Exception as e:
        logging.error("carl_progress v2: block hook failed: %r", e)

    # ---- hook 3b (v4.1): LTX transformer block forwards -------------------
    # The block-forward count IS the live unit (same shape as the Wan hook).
    # v4.1: LTX-2.5 22B loads as the A/V model — LTXAVModel(LTXVModel) with
    # its blocks = BasicAVTransformerBlock (comfy/ldm/lightricks/av_model.py),
    # so v4's single-class hook never fired (calls stayed 0, verified). Wrap
    # BOTH the video and AV block classes; only the one the model actually
    # instantiates will tick. Diagnostics live in the state file payload.
    try:
        import comfy.ldm.lightricks.model as ltxm
        _diag_note("ltx_block", imp=True)
        orig_ltx_fwd = ltxm.BasicTransformerBlock.forward

        def ltx_block_fwd(self, *a, **k):
            try:
                _diag_calls("ltx_block")
                j = _last_open()
                if j is not None:
                    j["blocks_done"] = int(j.get("blocks_done", 0)) + 1
                    now = time.time()
                    if now - j.get("_be", 0.0) >= 0.033:
                        j["_be"] = now
                        _emit()
            except Exception:
                pass
            return orig_ltx_fwd(self, *a, **k)
        ltxm.BasicTransformerBlock.forward = ltx_block_fwd
        _diag_note("ltx_block", installed=True)
        logging.info("carl_progress v4.1: ltx block hook (video class)")
    except Exception as e:
        _diag_note("ltx_block", first_err=repr(e)[:200])
        logging.error("carl_progress v4.1: ltx block hook failed: %r", e)

    try:
        import comfy.ldm.lightricks.av_model as ltxav
        _diag_note("ltx_av_block", imp=True)
        orig_ltxav_fwd = ltxav.BasicAVTransformerBlock.forward

        def ltxav_block_fwd(self, *a, **k):
            try:
                _diag_calls("ltx_av_block")
                j = _last_open()
                if j is not None:
                    j["blocks_done"] = int(j.get("blocks_done", 0)) + 1
                    now = time.time()
                    if now - j.get("_be", 0.0) >= 0.033:
                        j["_be"] = now
                        _emit()
            except Exception:
                pass
            return orig_ltxav_fwd(self, *a, **k)
        ltxav.BasicAVTransformerBlock.forward = ltxav_block_fwd
        _diag_note("ltx_av_block", installed=True)
        logging.info("carl_progress v4.1: ltx block hook (AV class)")
    except Exception as e:
        _diag_note("ltx_av_block", first_err=repr(e)[:200])
        logging.error("carl_progress v4.1: ltx AV block hook failed: %r", e)

    # ---- hook 3c (v4): LTX video VAE decode chunks ------------------------
    # LTX-2.5 video VAE (sd.py:749) = comfy.ldm.lightricks.vae.causal_video_autoencoder.
    # Decoder.forward_orig: torch.chunk(sample, num_chunks, dim=2) -> one
    # run_up() dispatch per chunk = the countable decode unit.
    try:
        import comfy.ldm.lightricks.vae.causal_video_autoencoder as ltxv
        orig_ltx_dec = ltxv.Decoder.forward_orig

        def ltx_dec_fwd(self, x, *a, **k):
            # forward_orig computes num_chunks internally AFTER our call —
            # so the real unit = run_up() dispatches (one per temporal chunk).
            # chunks_total stays 0 => the page uses its wall-clock ETA fallback
            # (already supported), chunks_done still drives the chunk counter.
            try:
                import torch
                if isinstance(x, torch.Tensor) and x.ndim >= 3:
                    orig_run_up = self.run_up

                    def cu_run_up(idx, *ra, **rk):
                        try:
                            with _lock:
                                dd = _live["decode"]
                                if dd is None or dd.get("state") == "done":
                                    _live["decode"] = {"state": "running", "chunks_done": 0,
                                                       "chunks_total": 0, "t0": time.time()}
                                    dd = _live["decode"]
                                dd["chunks_done"] = int(dd.get("chunks_done", 0)) + 1
                        except Exception:
                            pass
                        return orig_run_up(idx, *ra, **rk)
                    self.run_up = cu_run_up
                    out = orig_ltx_dec(self, x, *a, **k)
                    self.run_up = orig_run_up
                    return out
            except Exception:
                pass
            return orig_ltx_dec(self, x, *a, **k)
        ltxv.Decoder.forward_orig = ltx_dec_fwd
        logging.info("carl_progress v4: ltx vae hook")
    except Exception as e:
        logging.error("carl_progress v4: ltx vae hook failed: %r", e)

    # ---- hook 4: VAE decode chunks (BOTH Wan VAE classes) ------------------
    # Wan 2.1: comfy.ldm.wan.vae      — chunks_total = 1 + t//2
    # Wan 2.2: comfy.ldm.wan.vae2_2   — chunks_total = t (one Decoder3d.forward
    # per latent frame; 121f -> 31 chunks)   <- the class THIS stack loads
    # (comfy/sd.py:825 for wan2.2_vae). v2 wrapped only the 2.1 class: the
    # tail hooks never fired on any 2.2 render (A0M2/A0V/A0W) — fixed in v3.
    try:
        import comfy.ldm.wan.vae as wv

        dec_total = {"n": None, "t": 0.0}
        orig_dec = wv.Decoder3d.forward

        def dec_fwd(self, x, feat_cache=None, feat_idx=[0], **kw):
            try:
                # each call = one temporal chunk (WanVAE.decode loops the
                # decoder over t-chunks; counted unit is identical in both
                # 2.1 and 2.2)
                with _lock:
                    d = _live["decode"]
                    if d is None or d.get("state") == "done":
                        _live["decode"] = {"state": "running", "chunks_done": 0,
                                           "chunks_total": dec_total["n"] or 0, "t0": time.time()}
                        d = _live["decode"]
                    d["chunks_done"] = int(d.get("chunks_done", 0)) + 1
                    d["last_t"] = time.time()
                    tot = d.get("chunks_total") or 0
                    if tot and d["chunks_done"] >= tot:
                        d["state"] = "done"
                        d["finished"] = time.time()
                _emit()
            except Exception:
                pass
            return orig_dec(self, x, feat_cache=feat_cache, feat_idx=feat_idx, **kw)
        wv.Decoder3d.forward = dec_fwd

        orig_vae_dec = wv.WanVAE.decode

        def vae_decode(self, z, *a, **k):
            try:
                if isinstance(z, __import__("torch").Tensor) and z.ndim == 5:
                    t = int(z.shape[2])
                    with _lock:
                        dec_total["n"] = 1 + t // 2          # 2.1 math
                        dec_total["t"] = time.time()
                        if _live["decode"] is None or _live["decode"].get("state") == "done":
                            _live["decode"] = {"state": "running", "chunks_done": 0,
                                               "chunks_total": dec_total["n"], "t0": time.time()}
                    _emit()
            except Exception:
                pass
            return orig_vae_dec(self, z, *a, **k)
        wv.WanVAE.decode = vae_decode

        # --- Wan 2.2 module: same counting, 2.2 chunk math ---
        try:
            import comfy.ldm.wan.vae2_2 as wv2
            orig_dec22 = wv2.Decoder3d.forward

            def dec_fwd22(self, x, feat_cache=None, feat_idx=[0], **kw):
                try:
                    with _lock:
                        d = _live["decode"]
                        if d is None or d.get("state") == "done":
                            _live["decode"] = {"state": "running", "chunks_done": 0,
                                               "chunks_total": dec_total["n"] or 0, "t0": time.time()}
                            d = _live["decode"]
                        d["chunks_done"] = int(d.get("chunks_done", 0)) + 1
                        d["last_t"] = time.time()
                        tot = d.get("chunks_total") or 0
                        if tot and d["chunks_done"] >= tot:
                            d["state"] = "done"
                            d["finished"] = time.time()
                    _emit()
                except Exception:
                    pass
                return orig_dec22(self, x, feat_cache=feat_cache, feat_idx=feat_idx, **kw)
            wv2.Decoder3d.forward = dec_fwd22

            orig_vae_dec22 = wv2.WanVAE.decode

            def vae_decode22(self, z, *a, **k):
                try:
                    if isinstance(z, __import__("torch").Tensor) and z.ndim == 5:
                        t = int(z.shape[2])
                        with _lock:
                            dec_total["n"] = t               # 2.2: one chunk per latent frame
                            dec_total["t"] = time.time()
                            if _live["decode"] is None or _live["decode"].get("state") == "done":
                                _live["decode"] = {"state": "running", "chunks_done": 0,
                                                   "chunks_total": dec_total["n"], "t0": time.time()}
                        _emit()
                except Exception:
                    pass
                return orig_vae_dec22(self, z, *a, **k)
            wv2.WanVAE.decode = vae_decode22
            logging.info("carl_progress v2: vae2_2 hook (the 2.2 stack class)")
        except Exception:
            pass  # 2.2 module absent (older checkout) — 2.1 hooks stand
        logging.info("carl_progress v2: vae-decode hook")
    except Exception as e:
        logging.error("carl_progress v2: vae hook failed: %r", e)

    # ---- hook 5 (v4.1): per-node engine-truth units -----------------------
    # Every node of the active prompt counts; single-shot nodes are 1 unit
    # (true/false completed). Two independent wraps, each with its own
    # try + per-hook diag in the state file payload (v4's shared try let
    # one dead import kill both hooks silently — the 09-11 failure).
    #
    # Denominator: PromptQueue.put executed.py:1262 — server.py:1131 calls
    # self.prompt_queue.put((number, prompt_id, prompt, extra_data,
    # outputs_to_execute, sensitive)); item[1] = prompt_id, item[2] = the
    # node map, len(node_map) = total units. (This ComfyUI build has NO
    # queue_management module — it lives in execution.py.)
    #
    # Numerator: PromptServer.send_sync("executed", {"node", "display_node",
    # "output", "prompt_id"}, sid) — execution.py:578, the engine's own
    # per-node completion record.

    try:
        import execution
        _diag_note("node_put", imp=True)
        orig_put = execution.PromptQueue.put

        def pq_put(self, item, *a, **k):
            try:
                _diag_calls("node_put")
                pid = None
                pmap = None
                if isinstance(item, dict):
                    pid = item.get("prompt_id", item.get("id"))
                    pmap = item.get("prompt")
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    pid = item[1]
                    pmap = item[2]
                if pid is not None and isinstance(pmap, dict):
                    with _lock:
                        _cur_prompt[0] = str(pid)
                        _cur_prompt[1] = len(pmap)
                        _nodes_exec.clear()
                        if _exec_prompt[0] != str(pid):
                            _nodes_last.clear()
                    _emit()
            except Exception:
                pass
            return orig_put(self, item, *a, **k)
        execution.PromptQueue.put = pq_put
        _diag_note("node_put", installed=True)
        logging.info("carl_progress v4.1: prompt-put hook")
    except Exception as e:
        _diag_note("node_put", first_err=repr(e)[:200])
        logging.error("carl_progress v4.1: prompt-put hook failed: %r", e)

    try:
        from server import PromptServer
        _diag_note("node_executed", imp=True)
        orig_send = PromptServer.send_sync

        def ps_send(self, event, data, sid, *a, **k):
            try:
                if event == "executed" and isinstance(data, dict):
                    _diag_calls("node_executed")
                    pid = str(data.get("prompt_id", ""))
                    node_id = data.get("display_node", data.get("node"))
                    if node_id is not None and pid:
                        if _exec_prompt[0] != _cur_prompt[0]:
                            _exec_prompt[0] = str(pid)
                            _nodes_exec.clear()
                        if pid == _cur_prompt[0] and node_id not in _nodes_exec:
                            _nodes_exec.add(node_id)
                            if len(_nodes_exec) >= _cur_prompt[1]:
                                # close: every unit of this prompt completed —
                                # freeze as the last-closed result (the engine
                                # sends no success event to send_sync, so full
                                # count = authoritative close). Mutate in place
                                # (rebinding would need `global`); under _lock
                                # since _cur_nodes_ids reads it lock-free-safe.
                                _nodes_last.clear()
                                _nodes_last.update(_nodes_exec)
                            _emit()
            except Exception:
                pass
            return orig_send(self, event, data, sid, *a, **k)
        PromptServer.send_sync = ps_send
        _diag_note("node_executed", installed=True)
        logging.info("carl_progress v4.1: send_sync node hook")
    except Exception as e:
        _diag_note("node_executed", first_err=repr(e)[:200])
        logging.error("carl_progress v4.1: send_sync node hook failed: %r", e)

    _clip_monitor()  # encode-phase tracker (see below)


def _live_detail():
    """In-memory view of the whole chain — always complete.

    v3.2 (09-10): jobs are SHALLOW-COPIED before private keys are stripped.
    v3.1 popped `_step_now`/`_be` off the LIVE dicts (list() copies the list,
    not the elements) — every REST read mid-sampling reset the per-step
    anchor map, s_per_step re-anchored to the next callback (A0X: 163/70 s,
    A0Y: 271 then 338 s — the '70s/step' phantom). _emit already copied
    correctly; this route did not."""
    with _lock:
        detail = {
            "ts": time.time(),
            "jobs": [dict(j) for j in sorted(_live["jobs"], key=lambda j: j.get("_started", 0))],
            "history": list(_live["history"]),
            "decode": dict(_live["decode"]) if _live["decode"] else None,
            "encode": dict(_live["encode"]) if _live["encode"] else None,
            "nodes": {"total": int(_cur_prompt[1]), "done": _nodes_done_n(),
                      "ids": sorted(_cur_nodes_ids()) if _cur_prompt[0] is not None else [],
                      "prompt": _cur_prompt[0]},
            # v4.1: per-hook install/fire diagnostics (snapshot; takes _diag_lock).
            "diag": _diag_snapshot(),
        }
        for j in detail["jobs"]:
            j.pop("_last_t", None)
            j.pop("_be", None)
            j.pop("_step_now", None)
            j.pop("_hc", None)
            j.pop("_hs", None)
            j.pop("_hn", None)
    d = detail.get("decode")
    if d and d.get("state") == "running":
        tot = d.get("chunks_total") or 0
        d["pct"] = round(100.0 * d["chunks_done"] / tot, 1) if tot else None
        d["elapsed"] = round(time.time() - d["t0"], 1)
        per = max(0.1, d["elapsed"] / max(1, d["chunks_done"]))
        left = max(0, tot - d["chunks_done"])
        d["eta_s"] = int(left * per) if tot else None
        d["s_per_chunk"] = round(per, 2)
    return detail


def _reset_arm():
    """Arm the chain for the NEXT render: clear phase state, mark the newest
    existing mp4 as already-seen so only a genuinely new file trips encode."""
    global _prev_mp4_ts
    with _lock:
        _live["decode"] = None
        _live["encode"] = None
        _prev_mp4_ts = _max_mp4_mtime()
    _emit()


def _clip_monitor():
    """Detect the moment the output mp4 appears/grows (CreateVideo+SaveVideo
    phase). One daemon thread, 2 s cadence, writes only the state file.

    State machine:
      idle     -> no open jobs, decode/encode not running (settled 5 ticks)
                 -> _reset_arm() so the next render's new mp4 is what trips encode
      encode   -> newest mp4 (mt > armed marker) appears / grows; size stops for
                 ~4 s  -> encode done -> back to idle
    """
    if not ENABLED:
        return

    def run():
        while True:
            try:
                with _lock:
                    any_open = any(j.get("state") != "done" for j in _live["jobs"])
                    dec = dict(_live["decode"]) if _live["decode"] else None
                    enc = dict(_live["encode"]) if _live["encode"] else None
                    marker = _prev_mp4_ts

                # --- arm window: no encode phase yet, and a chain is in flight
                #     (sampler open, or a decode phase exists = we're inside a
                #     render). Marker guard: only an mp4 NEWER than the armed
                #     snapshot trips encode, so old files never re-trigger. ---
                if enc is None and (any_open or dec is not None):
                    mp4 = _find_new_mp4(marker or 0)
                    if mp4 is not None:
                        with _lock:
                            if _live["encode"] is None or _live["encode"].get("state") == "done":
                                _live["encode"] = {"state": "running", "started": time.time()}
                            _live["encode"]["clip_bytes"] = mp4["bytes"]
                            _live["encode"]["clip_file"] = mp4["file"]
                            _live["encode"]["clip_mt"] = mp4["mt"]
                        _emit()

                # --- encode growth / completion ---
                run_enc = _live["encode"] is not None and _live["encode"].get("state") == "running"
                if run_enc:
                    mp4 = _find_new_mp4(0)
                    if mp4:
                        grow = False
                        with _lock:
                            cur = _live["encode"]
                            if cur is not None and cur.get("state") == "running":
                                if mp4["bytes"] > cur.get("clip_bytes", 0):
                                    cur["clip_bytes"] = mp4["bytes"]
                                    cur["clip_file"] = mp4["file"]
                                    grow = True
                                cur["elapsed"] = round(time.time() - cur["started"], 1)
                        if grow:
                            with _lock:
                                _enc_idle[0] = 0
                        else:
                            with _lock:
                                _enc_idle[0] += 1
                        with _lock:
                            if _live["encode"] and _live["encode"].get("state") == "running" \
                                    and _enc_idle[0] >= 2:  # ~4 s stable
                                _live["encode"]["state"] = "done"
                                _live["encode"]["finished"] = time.time()
                                _live["encode"]["total_bytes"] = mp4["bytes"]
                        _emit()

                # --- idle reset: re-arm for the next render ---
                enc_done = _live["encode"] is not None
                enc_run = _live["encode"] is not None and _live["encode"].get("state") == "running"
                dec_run = _live["decode"] is not None and _live["decode"].get("state") == "running"
                if not any_open and not dec_run and not enc_run:
                    with _lock:
                        _idle[0] += 1
                    if _idle[0] >= 5 and enc_done:  # ~10 s fully settled
                        _reset_arm()
                        with _lock:
                            _idle[0] = 0
                else:
                    with _lock:
                        _idle[0] = 0
            except Exception:
                pass
            time.sleep(2)

    t = threading.Thread(target=run, name="carl-clip-monitor", daemon=True)
    t.start()


def _install_route():
    global _LOOP
    try:
        if _WEB_OK:
            from server import PromptServer
            import asyncio as _aio
            ps = getattr(PromptServer, "instance", None)
            if ps is not None:
                try:
                    _LOOP = _aio.get_running_loop()
                except Exception:
                    _LOOP = None
                # route install runs at IMPORT time — the server loop does not
                # exist yet; _install_route captures what it can, and the WS
                # handler below (first client) captures the real one.

                @ps.routes.get("/carl/progress")
                async def carl_progress(request):
                    try:
                        detail = _live_detail()
                    except Exception:
                        detail = None
                        try:
                            with open(STATE_PATH, "r", encoding="utf-8") as f:
                                detail = json.load(f)
                        except Exception:
                            detail = {"ts": time.time(), "jobs": [], "history": []}
                    state = _state_of(detail)
                    return web.json_response({"state": state, "detail": detail})

                # Real-time push (v3): the 666 server keeps one persistent
                # ssh leg running carl_ws_tail.py on the box as a WS client;
                # every _emit() is pushed here the instant it happens —
                # block-by-block, chunk-by-chunk, no poll interval anywhere.
                @ps.routes.get("/carl/progress/ws")
                async def carl_progress_ws(request):
                    global _LOOP
                    try:
                        _LOOP = _aio.get_running_loop()
                    except Exception:
                        pass
                    ws = web.WebSocketResponse(heartbeat=15.0)
                    await ws.prepare(request)
                    _WS_CLIENTS.add(ws)
                    try:
                        detail = _live_detail()
                        res = ws.send_str(json.dumps({"state": _state_of(detail), "detail": detail}))
                        if res is not None:
                            await res
                    except Exception:
                        pass
                    try:
                        async for _m in ws:
                            pass  # one-way push; client is silent
                    except Exception:
                        pass
                    finally:
                        _WS_CLIENTS.discard(ws)
                    return ws

                logging.info("carl_progress v2: route /carl/progress + /carl/progress/ws registered, state %s", STATE_PATH)
            else:
                logging.warning("carl_progress v2: PromptServer.instance unavailable, hooks active, no route yet")
        else:
            logging.warning("carl_progress v2: aiohttp import failed, hooks active, no route")
    except Exception as e:
        logging.error("carl_progress v2: route install failed: %r", e)


patch()
_install_route()
