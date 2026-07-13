"""
AceStep Worker - runs as a standalone Flask server on port 5001.
StemSplit proxies AI remix requests here.
Started automatically by start_all.bat
"""
import sys, os, uuid, threading, argparse, gc, dataclasses
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

gen_jobs          = {}
dit_handler       = None
llm_handler       = None
ready             = False
loading           = True
load_error        = ''
cpu_inference_mode = False   # set by --cpu-inference flag


# ── MODEL INIT ───────────────────────────────────────────────────────
def init_models(acestep_dir):
    global dit_handler, llm_handler, ready, loading, load_error, cpu_inference_mode
    try:
        print('[Worker] Importing AceStep...')
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        import torch

        if cpu_inference_mode:
            device = 'cpu'
            print('[Worker] CPU inference mode (--cpu-inference flag) — models load into RAM, no VRAM limit')
        elif torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free_gb    = torch.cuda.mem_get_info()[0] / (1024**3)
            print(f'[Worker] GPU: {torch.cuda.get_device_name(0)}, '
                  f'Total VRAM: {vram_total:.1f} GB, Free: {free_gb:.1f} GB')
            # ACE-Step models consume 5-6 GB VRAM; inference needs ~1 GB on top.
            # Require ≥7 GB free *before* loading so there is headroom after.
            # Checking free_gb (not total) catches GPUs where other apps already
            # consumed most VRAM (e.g. 6 GB card with 5.8 GB free → 0.2 GB left
            # after loading → inference OOM regardless of card size).
            if free_gb < 7.0:
                cpu_inference_mode = True
                device = 'cpu'
                print(f'[Worker] Auto-switching to CPU: only {free_gb:.1f} GB VRAM free '
                      f'(need ~7 GB — ~6 GB for models + ~1 GB for inference). '
                      f'Generation will be slower but has no VRAM limit.')
            else:
                device = 'cuda'
        else:
            device = 'cpu'

        print('[Worker] Initializing DiT model (downloading ~5GB on first run)...')
        dit_handler = AceStepHandler()
        dit_handler.initialize_service(
            project_root=acestep_dir,
            config_path='acestep-v15-turbo',
            device=device
        )

        print('[Worker] Initializing LM model...')
        llm_handler = LLMHandler()
        llm_handler.initialize(
            checkpoint_dir=os.path.join(acestep_dir, 'checkpoints'),
            lm_model_path='acestep-5Hz-lm-0.6B',
            backend='pt',
            device=device
        )

        if device == 'cuda':
            free_after = torch.cuda.mem_get_info()[0] / (1024**3)
            print(f'[Worker] VRAM after loading: {free_after:.1f} GB free of {vram_total:.1f} GB total')
        else:
            print('[Worker] Models loaded on CPU.')

        ready   = True
        loading = False
        print('[Worker] Models ready!')

    except Exception as e:
        import traceback; traceback.print_exc()
        load_error = str(e)
        loading    = False
        ready      = False
        print(f'[Worker] ERROR: {e}')


# ── VRAM-SAFE GENERATE ───────────────────────────────────────────────
def _is_vram_error(e):
    s = str(e).lower()
    return any(k in s for k in ['vram', 'out of memory', 'cuda out', 'insufficient'])

def _is_vram_result(result):
    """ACE-Step preflight returns a failed result (not an exception) on VRAM shortage."""
    if getattr(result, 'success', True):
        return False
    msg = str(getattr(result, 'status_message', '') or '').lower()
    return 'insufficient free vram' in msg or 'vram' in msg

def _trim_to_vram_limit(src_path, out_dir, guidance_scale=1.0):
    """
    Trim src_path to the max single-pass duration that fits in current free VRAM.
    Returns (path, was_trimmed, duration_s, free_gb).
    Raises RuntimeError with actionable message if VRAM is too low to even attempt.

    Margin rationale: ACE-Step uses 0.5 GB internally; empirically the actual
    peak runs ~0.3 GB higher due to audio encoding and intermediate buffers that
    aren't captured in the per-batch estimate. We use 0.8 GB total margin.
    """
    import torch
    from pydub import AudioSegment

    # CPU mode has no VRAM constraint — pass full audio through unchanged
    if cpu_inference_mode or not torch.cuda.is_available():
        from pydub import AudioSegment
        audio = AudioSegment.from_file(src_path)
        return src_path, False, len(audio) / 1000.0, 0.0

    try:
        torch.cuda.empty_cache()
        gc.collect()
        free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
    except Exception:
        free_gb = 999.0

    per_batch = 0.6 if guidance_scale > 1.0 else 0.3
    max_dur   = (free_gb - 0.8) / per_batch * 60.0
    max_dur   = max(0.0, min(max_dur, 240.0))

    min_useful = 15.0
    if max_dur < min_useful:
        needed = round(0.8 + per_batch * (min_useful / 60), 1)
        raise RuntimeError(
            f'Not enough free VRAM for AI generation: {free_gb:.2f} GB free, '
            f'~{needed} GB needed for even a {min_useful:.0f}s clip. '
            f'Run stop_all.bat → start_all.bat to reclaim GPU memory, '
            f'or close other GPU-intensive apps first.'
        )

    audio   = AudioSegment.from_file(src_path)
    total_s = len(audio) / 1000.0
    if total_s <= max_dur:
        return src_path, False, total_s, free_gb

    trimmed   = audio[: int(max_dur * 1000)]
    trim_path = os.path.join(out_dir, '_vram_trimmed.mp3')
    trimmed.export(trim_path, format='mp3', parameters=['-q:a', '2'])
    return trim_path, True, max_dur, free_gb

def vram_safe_generate(gp, gc_config, save_dir, job_id=None):
    """
    Calls generate_music after clearing cached VRAM.
    On VRAM error gives a clear actionable message — does NOT retry
    (retries can hang if GPU is in bad state after a CUDA error).
    """
    from acestep.inference import generate_music
    import torch

    # Clear cached (unused) VRAM — do NOT call synchronize() here,
    # it can hang indefinitely if the GPU is in an error state.
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    try:
        result = generate_music(dit_handler, llm_handler, gp, gc_config, save_dir=save_dir)
        if _is_vram_result(result):
            # Preflight returned a structured VRAM-failure result — surface as RuntimeError
            raise RuntimeError(getattr(result, 'status_message', 'Insufficient VRAM'))
        return result
    except RuntimeError as e:
        if not _is_vram_error(e):
            raise  # Not a VRAM problem, bubble up as-is
        dur = getattr(gp, 'duration', '?')
        try:
            free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
            detail  = f'{free_gb:.1f} GB free VRAM'
        except Exception:
            detail = 'limited VRAM'
        raise RuntimeError(
            f'Not enough VRAM to generate {dur}s of audio ({detail}). '
            f'Reduce Duration to 60s or less and try again. '
            f'If the problem persists, restart start_all.bat to clear GPU memory.'
        )



# ── SHARED HELPERS ───────────────────────────────────────────────────
def new_job():
    jid = str(uuid.uuid4())[:10]
    gen_jobs[jid] = {'status': 'processing', 'message': 'Queued...', 'path': None}
    return jid

def job_done(jid, path):
    gen_jobs[jid]['status']  = 'done'
    gen_jobs[jid]['path']    = os.path.abspath(path)
    gen_jobs[jid]['message'] = 'Complete!'

def job_err(jid, msg):
    gen_jobs[jid]['status']  = 'error'
    gen_jobs[jid]['message'] = msg


# ── ROUTES ───────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({'ready': ready, 'loading': loading, 'error': load_error})


# ── 1. GENERATE FRESH ────────────────────────────────────────────────
@app.route('/generate', methods=['POST'])
def generate():
    if not ready:
        return jsonify({'error': 'Models still loading...' if loading else load_error}), 503
    data = request.get_json() or {}
    jid  = new_job()
    t = threading.Thread(target=run_generate, args=(jid, data))
    t.daemon = True; t.start()
    return jsonify({'job_id': jid})

def run_generate(jid, params):
    try:
        from acestep.inference import GenerationParams, GenerationConfig
        gen_jobs[jid]['message'] = 'Generating music...'
        out_dir = os.path.abspath(os.path.join('outputs', 'ai_remix', jid))
        os.makedirs(out_dir, exist_ok=True)
        gp_kw = {
            'task_type': 'text2music',
            'caption':   params.get('prompt', 'instrumental music'),
            'duration':  min(max(float(params.get('duration', 60)), 5), 240),
        }
        if params.get('bpm'):
            try: gp_kw['bpm'] = int(params['bpm'])
            except Exception: pass
        gp = GenerationParams(**gp_kw)
        result = vram_safe_generate(gp, GenerationConfig(batch_size=1, audio_format='mp3'), out_dir, jid)
        if result.success and result.audios:
            job_done(jid, result.audios[0]['path'])
        else:
            job_err(jid, getattr(result, 'error', 'Generation failed'))
    except Exception as e:
        import traceback; traceback.print_exc()
        job_err(jid, str(e))


# ── 2. STYLE TRANSFER (cover) ────────────────────────────────────────
@app.route('/cover', methods=['POST'])
def cover():
    if not ready:
        return jsonify({'error': 'Models still loading...' if loading else load_error}), 503
    data = request.get_json() or {}
    src = data.get('src_audio_path', '')
    if not src or not os.path.exists(src):
        return jsonify({'error': 'Source audio file not found on disk'}), 400
    jid = new_job()
    t = threading.Thread(target=run_cover, args=(jid, data))
    t.daemon = True; t.start()
    return jsonify({'job_id': jid})

def run_cover(jid, params):
    try:
        from acestep.inference import GenerationParams, GenerationConfig
        src_path  = params['src_audio_path']
        out_dir   = os.path.abspath(os.path.join('outputs', 'ai_remix', jid))
        os.makedirs(out_dir, exist_ok=True)
        strength   = float(params.get('strength', 0.4))
        orig_dur   = float(params.get('duration', 60.0))
        gc_config  = GenerationConfig(batch_size=1, audio_format='mp3')
        COVER_GUIDANCE = 9.0

        src_path, was_trimmed, duration, free_gb = _trim_to_vram_limit(src_path, out_dir, guidance_scale=COVER_GUIDANCE)
        if was_trimmed:
            gen_jobs[jid]['message'] = (
                f'Limited VRAM ({free_gb:.1f} GB free) — generating first '
                f'{duration:.0f}s of {orig_dur:.0f}s track...'
            )
        else:
            gen_jobs[jid]['message'] = 'Analyzing your track and transferring style...'

        gp_kw = {
            'task_type':            'cover',
            'src_audio':            src_path,
            'caption':              params.get('prompt', ''),
            'audio_cover_strength': strength,
            'duration':             min(max(duration, 5.0), 240.0),
            'inference_steps':      28,
            'guidance_scale':       COVER_GUIDANCE,
            'shift':                3.0,
            'cfg_interval_start':   0.0,
            'cfg_interval_end':     0.95,
            'infer_method':         'ode',
        }
        try:
            gp = GenerationParams(**gp_kw)
        except TypeError:
            gp = GenerationParams(task_type='cover', src_audio=src_path,
                                  caption=params.get('prompt', ''),
                                  duration=gp_kw['duration'])
        result = vram_safe_generate(gp, gc_config, out_dir, jid)
        if result.success and result.audios:
            job_done(jid, result.audios[0]['path'])
        else:
            job_err(jid, getattr(result, 'status_message', 'Style transfer failed'))
    except Exception as e:
        import traceback; traceback.print_exc()
        job_err(jid, str(e))


# ── 3. REFERENCE STYLE ───────────────────────────────────────────────
@app.route('/reference', methods=['POST'])
def reference():
    if not ready:
        return jsonify({'error': 'Models still loading...' if loading else load_error}), 503
    data = request.get_json() or {}
    ref = data.get('reference_audio_path', '')
    if not ref or not os.path.exists(ref):
        return jsonify({'error': 'Reference audio file not found'}), 400
    jid = new_job()
    t = threading.Thread(target=run_reference, args=(jid, data))
    t.daemon = True; t.start()
    return jsonify({'job_id': jid})

def run_reference(jid, params):
    try:
        from acestep.inference import GenerationParams, GenerationConfig
        gen_jobs[jid]['message'] = 'Generating music with your track as acoustic reference...'
        ref_path = params['reference_audio_path']
        out_dir  = os.path.abspath(os.path.join('outputs', 'ai_remix', jid))
        os.makedirs(out_dir, exist_ok=True)
        gp_kw = {
            'task_type':       'text2music',
            'reference_audio': ref_path,
            'caption':         params.get('prompt', 'instrumental music'),
            'duration':        min(max(float(params.get('duration', 60)), 5), 240),
        }
        if params.get('bpm'):
            try: gp_kw['bpm'] = int(params['bpm'])
            except Exception: pass
        try:
            gp = GenerationParams(**gp_kw)
        except TypeError:
            gp = GenerationParams(task_type='text2music', caption=gp_kw['caption'],
                                  duration=gp_kw['duration'])
        result = vram_safe_generate(gp, GenerationConfig(batch_size=1, audio_format='mp3'), out_dir, jid)
        if result.success and result.audios:
            job_done(jid, result.audios[0]['path'])
        else:
            job_err(jid, getattr(result, 'error', 'Generation failed'))
    except Exception as e:
        import traceback; traceback.print_exc()
        job_err(jid, str(e))


# ── 4. STEM REPLACEMENT ──────────────────────────────────────────────
@app.route('/replace', methods=['POST'])
def replace_stem():
    if not ready:
        return jsonify({'error': 'Models still loading...' if loading else load_error}), 503
    data = request.get_json() or {}
    ctx  = data.get('context_audio_path', '')
    if not ctx or not os.path.exists(ctx):
        return jsonify({'error': 'Context audio file not found on disk'}), 400
    jid = new_job()
    t = threading.Thread(target=run_replace, args=(jid, data))
    t.daemon = True; t.start()
    return jsonify({'job_id': jid})

def run_replace(jid, params):
    try:
        from acestep.inference import GenerationParams, GenerationConfig
        ctx_path  = params['context_audio_path']
        prompt    = params.get('prompt', 'instrumental music')
        orig_dur  = float(params.get('duration', 60.0))
        stem_name = params.get('stem_name', 'instrument')
        out_dir   = os.path.abspath(os.path.join('outputs', 'ai_remix', jid))
        os.makedirs(out_dir, exist_ok=True)
        gc_config = GenerationConfig(batch_size=1, audio_format='mp3')

        # lego task runs in turbo mode (no CFG, 0.3 GB/batch)
        ctx_path, was_trimmed, duration, free_gb = _trim_to_vram_limit(ctx_path, out_dir, guidance_scale=1.0)
        if was_trimmed:
            gen_jobs[jid]['message'] = (
                f'Limited VRAM ({free_gb:.1f} GB free) — generating first '
                f'{duration:.0f}s of {orig_dur:.0f}s track...'
            )
        else:
            gen_jobs[jid]['message'] = f'Generating {stem_name} in musical context...'

        try:
            gp = GenerationParams(
                task_type='lego',
                src_audio=ctx_path,
                instruction=f'Generate the {stem_name} track based on the audio context:',
                caption=prompt,
                duration=min(max(duration, 5.0), 240.0),
            )
        except TypeError:
            gp = GenerationParams(
                task_type='text2music',
                reference_audio=ctx_path,
                caption=prompt,
                duration=min(max(duration, 5.0), 240.0),
            )
        result = vram_safe_generate(gp, gc_config, out_dir, jid)
        if result.success and result.audios:
            job_done(jid, result.audios[0]['path'])
        else:
            job_err(jid, getattr(result, 'status_message', 'Generation failed'))
    except Exception as e:
        import traceback; traceback.print_exc()
        job_err(jid, str(e))


# ── STATUS & DOWNLOAD ────────────────────────────────────────────────
@app.route('/job/<jid>')
def job_status(jid):
    job = gen_jobs.get(jid)
    if not job: return jsonify({'status': 'not_found'}), 404
    return jsonify({'status': job['status'], 'message': job['message']})

@app.route('/download/<jid>')
def download(jid):
    job = gen_jobs.get(jid)
    if not job or job['status'] != 'done' or not job['path']:
        return jsonify({'error': 'Not ready'}), 404
    if not os.path.exists(job['path']):
        return jsonify({'error': 'File missing'}), 404
    return send_file(job['path'], as_attachment=True,
                     download_name='ai_remix.mp3', mimetype='audio/mpeg')


# ── ENTRY POINT ──────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--acestep-dir', default=os.path.abspath('.'))
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--cpu-inference', action='store_true',
                        help='Load models on CPU instead of GPU. Slower but no VRAM limit.')
    args = parser.parse_args()

    if args.cpu_inference:
        cpu_inference_mode = True

    adir = os.path.abspath(args.acestep_dir)
    if not os.path.isdir(adir):
        print(f'ERROR: AceStep directory not found: {adir}'); sys.exit(1)

    os.makedirs(os.path.join('outputs', 'ai_remix'), exist_ok=True)

    print(f'\n{"="*52}')
    print(f'  AceStep Worker  |  port {args.port}')
    if cpu_inference_mode:
        print(f'  Mode: CPU inference (no VRAM limit, slower)')
    print(f'{"="*52}')
    print(f'  Dir: {adir}')
    print(f'  Modes: generate / cover / reference / replace\n')

    t = threading.Thread(target=init_models, args=(adir,))
    t.daemon = True; t.start()

    app.run(debug=False, port=args.port, host='127.0.0.1')
