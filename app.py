import sys, os, uuid, glob, threading, subprocess, math, json as _json, re
import urllib.request, urllib.error
from flask import Flask, request, jsonify, send_file, Response

app = Flask(__name__)
UPLOAD_DIR = 'uploads'
OUTPUT_DIR = 'outputs'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
YT_DIR = 'yt_downloads'
os.makedirs(YT_DIR, exist_ok=True)
jobs    = {}
yt_jobs      = {}
replace_jobs = {}
split_jobs   = {}
WORKER = 'http://127.0.0.1:5001'

# ── WORKER PROXY ─────────────────────────────────────────────────────
def w_get(path):
    try:
        with urllib.request.urlopen(WORKER+path, timeout=4) as r:
            return _json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return _json.loads(e.read() or b'{}'), e.code
    except Exception as e:
        return {'error':str(e),'ready':False,'loading':False}, 503

def w_post(path, data):
    try:
        payload=_json.dumps(data).encode()
        req=urllib.request.Request(WORKER+path,data=payload,
            headers={'Content-Type':'application/json'},method='POST')
        with urllib.request.urlopen(req,timeout=10) as r:
            return _json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return _json.loads(e.read() or b'{}'), e.code
    except Exception as e:
        return {'error':str(e)}, 503

@app.route('/api/ai/status')
def ai_status(): d,c=w_get('/health'); return jsonify(d),c

@app.route('/api/ai/generate',methods=['POST'])
def ai_generate(): d,c=w_post('/generate',request.get_json() or {}); return jsonify(d),c

@app.route('/api/ai/cover',methods=['POST'])
def ai_cover():
    body=request.get_json() or {}
    sjid=body.get('stem_job_id')
    if not sjid: return jsonify({'error':'No stem job ID'}),400
    job=jobs.get(sjid)
    if not job: return jsonify({'error':'Track job not found'}),404
    up=job.get('upload_path','')
    if not up or not os.path.exists(up): return jsonify({'error':'Original audio not found'}),404
    d,c=w_post('/cover',{'src_audio_path':up,'prompt':body.get('prompt',''),'strength':body.get('strength',0.4)})
    return jsonify(d),c

@app.route('/api/ai/reference',methods=['POST'])
def ai_reference():
    body=request.get_json() or {}
    sjid=body.get('stem_job_id')
    if not sjid: return jsonify({'error':'No stem job ID'}),400
    job=jobs.get(sjid)
    if not job: return jsonify({'error':'Track job not found'}),404
    up=job.get('upload_path','')
    if not up or not os.path.exists(up): return jsonify({'error':'Original audio not found'}),404
    d,c=w_post('/reference',{'reference_audio_path':up,'prompt':body.get('prompt',''),
        'duration':body.get('duration',30),'bpm':body.get('bpm')})
    return jsonify(d),c

@app.route('/api/ai/job/<jid>')
def ai_job(jid): d,c=w_get(f'/job/{jid}'); return jsonify(d),c

@app.route('/api/ai/download/<jid>')
def ai_download(jid):
    try:
        with urllib.request.urlopen(f'{WORKER}/download/{jid}',timeout=15) as r:
            return Response(r.read(),mimetype='audio/mpeg',
                headers={'Content-Disposition':'attachment; filename=ai_remix.mp3'})
    except Exception as e:
        return jsonify({'error':str(e)}),503

# ── CLEAN & MASTER ────────────────────────────────────────────────────
INTENSITY = {
    'light':  {'nr':0.30,'gate':-62,'ct':-20,'cr':2.0,'ca':10,'cr2':150,'hm':0.85},
    'medium': {'nr':0.55,'gate':-52,'ct':-16,'cr':3.5,'ca':5, 'cr2':100,'hm':1.0},
    'heavy':  {'nr':0.80,'gate':-44,'ct':-12,'cr':6.0,'ca':2, 'cr2':60, 'hm':1.2},
}

PROFILES = {
    'vocals':    {'hp':90,  'lp':None,'ls':None,       'hs':(8000, 1.5)},
    'drums':     {'hp':40,  'lp':None,'ls':(80,  2.0), 'hs':(10000,-2.5)},
    'bass':      {'hp':30,  'lp':500, 'ls':(120, 1.5), 'hs':None},
    'guitar':    {'hp':80,  'lp':None,'ls':None,       'hs':(6000, 1.0)},
    'piano':     {'hp':60,  'lp':None,'ls':None,       'hs':(9000, 0.5)},
    'other':     {'hp':80,  'lp':None,'ls':None,       'hs':None},
    'no_vocals': {'hp':40,  'lp':None,'ls':None,       'hs':None},
}

def clean_one_stem(name, in_path, out_dir, p):
    import numpy as np
    from pedalboard.io import AudioFile
    from pedalboard import (Pedalboard, Compressor, Limiter,
                             HighpassFilter, LowpassFilter,
                             NoiseGate, HighShelfFilter, LowShelfFilter)
    from pydub import AudioSegment

    # Load
    with AudioFile(in_path) as f:
        audio = f.read(f.frames).astype(np.float32)
        sr    = int(f.samplerate)

    # Noise reduction
    try:
        import noisereduce as nr
        if audio.ndim == 2:
            reduced = np.stack([
                nr.reduce_noise(y=audio[c], sr=sr,
                                prop_decrease=p['nr'], stationary=False, n_jobs=1)
                for c in range(audio.shape[0])
            ])
        else:
            reduced = nr.reduce_noise(y=audio, sr=sr,
                                      prop_decrease=p['nr'], stationary=False)
            reduced = reduced[np.newaxis, :]
        # Pad if noisereduce shortened by a few samples
        if reduced.shape[-1] < audio.shape[-1]:
            pad = audio.shape[-1] - reduced.shape[-1]
            reduced = np.pad(reduced, ((0,0),(0,pad)))
        audio = reduced.astype(np.float32)
    except Exception:
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

    # Pedalboard chain
    prof  = PROFILES.get(name, PROFILES['other'])
    chain = []
    chain.append(NoiseGate(threshold_db=p['gate'], ratio=4.0,
                            attack_ms=2.0, release_ms=100.0))
    chain.append(HighpassFilter(cutoff_frequency_hz=prof['hp']*p['hm']))
    if prof['lp']:
        chain.append(LowpassFilter(cutoff_frequency_hz=prof['lp']))
    if prof['ls']:
        chain.append(LowShelfFilter(cutoff_frequency_hz=prof['ls'][0],
                                     gain_db=prof['ls'][1]))
    if prof['hs']:
        chain.append(HighShelfFilter(cutoff_frequency_hz=prof['hs'][0],
                                      gain_db=prof['hs'][1]))
    chain.append(Compressor(threshold_db=p['ct'], ratio=p['cr'],
                             attack_ms=p['ca'], release_ms=p['cr2']))
    chain.append(Limiter(threshold_db=-2.0, release_ms=100.0))
    processed = Pedalboard(chain)(audio, sr)

    # Loudness normalization
    try:
        import pyloudnorm as pyln
        import numpy as np
        data_t = processed.T.astype(np.float64)
        meter   = pyln.Meter(sr)
        lufs    = meter.integrated_loudness(data_t)
        if lufs > -70:
            data_t   = pyln.normalize.loudness(data_t, lufs, -16.0)
            processed = np.clip(data_t.T, -0.99, 0.99).astype(np.float32)
    except Exception:
        pass

    processed = np.clip(processed, -0.99, 0.99)

    # Save via pydub
    out_path  = os.path.join(out_dir, name+'.mp3')
    int16     = (processed * 32767).astype('int16')
    n_ch      = int16.shape[0] if int16.ndim == 2 else 1
    raw_bytes = int16.T.flatten().tobytes() if n_ch > 1 else int16.flatten().tobytes()
    AudioSegment(raw_bytes, frame_rate=sr, sample_width=2,
                 channels=n_ch).export(out_path, format='mp3', bitrate='320k')
    return out_path

def run_clean(jid, intensity):
    ck = jid+'_clean'
    try:
        job = jobs[jid]
        p   = INTENSITY.get(intensity, INTENSITY['medium'])
        od  = os.path.join(OUTPUT_DIR, jid, 'cleaned')
        os.makedirs(od, exist_ok=True)
        cleaned = []
        stems   = list(job['stems'])
        for i, stem in enumerate(stems):
            n = stem['name']
            jobs[ck]['message'] = f'Cleaning {n} ({i+1}/{len(stems)})...'
            try:
                cp = clean_one_stem(n, stem['path'], od, p)
                cleaned.append({'name':n,'path':cp})
            except Exception as e:
                print(f'[Clean] stem {n} failed: {e}')
                cleaned.append(stem)   # fall back to original
        job['stems']   = cleaned
        job['cleaned'] = True
        jobs[ck]['status']  = 'done'
        jobs[ck]['message'] = 'All stems cleaned!'
    except ImportError as e:
        jobs[ck]['status']  = 'error'
        jobs[ck]['message'] = f'Missing library: {e}. Run setup.bat to install pedalboard, noisereduce, pyloudnorm.'
    except Exception as e:
        import traceback; traceback.print_exc()
        jobs[ck]['status']  = 'error'
        jobs[ck]['message'] = str(e)

@app.route('/api/clean/<jid>', methods=['POST'])
def clean(jid):
    job = jobs.get(jid)
    if not job or job['status'] != 'done':
        return jsonify({'error':'Job not ready'}), 404
    data = request.get_json() or {}
    intensity = data.get('intensity','medium')
    ck = jid+'_clean'
    if ck in jobs and jobs[ck].get('status') == 'processing':
        return jsonify({'error':'Already processing'}), 409
    jobs[ck] = {'status':'processing','message':'Starting cleanup...'}
    t = threading.Thread(target=run_clean, args=(jid, intensity))
    t.daemon = True; t.start()
    return jsonify({'ok': True})

@app.route('/api/clean_status/<jid>')
def clean_status(jid):
    job = jobs.get(jid+'_clean')
    if not job: return jsonify({'status':'not_found'}),404
    return jsonify(job)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>StemSplit</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://unpkg.com/wavesurfer.js@6/dist/wavesurfer.min.js"></script>
  <script src="https://unpkg.com/wavesurfer.js@6/dist/plugin/wavesurfer.regions.min.js"></script>
  <script src="https://unpkg.com/wavesurfer.js@6/dist/plugin/wavesurfer.timeline.min.js"></script>
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    :root{
      --bg:#08080F;--surface:#0D0D1A;--card:#131325;--card2:#181832;
      --bdr:rgba(124,58,237,.35);--bds:rgba(255,255,255,.07);
      --pur:#7C3AED;--purl:#A78BFA;--purd:rgba(124,58,237,.12);
      --t:#F0F0FF;--t2:#9A9ABF;--t3:#5A5A7A;
      --grn:#10B981;--ylw:#F59E0B;--red:#F87171;
      --r:16px;--rs:10px
    }
    body{background:var(--bg);color:var(--t);font-family:"Inter",system-ui,sans-serif;min-height:100vh;display:flex;flex-direction:column;align-items:center}
    header{width:100%;padding:16px 36px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--bds);background:rgba(8,8,15,.9);backdrop-filter:blur(12px);position:sticky;top:0;z-index:50}
    .logo{display:flex;align-items:center;gap:10px}
    .logo-i{width:34px;height:34px;background:linear-gradient(135deg,#7C3AED,#A78BFA);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:17px}
    .logo-t{font-size:19px;font-weight:700}.logo-t span{color:var(--purl)}
    .badge{font-size:11px;font-weight:600;color:var(--purl);background:var(--purd);border:1px solid var(--bdr);padding:4px 11px;border-radius:20px;text-transform:uppercase;letter-spacing:.5px}
    main{width:100%;max-width:800px;padding:48px 20px 80px;display:flex;flex-direction:column;align-items:center;gap:22px}
    .hero{text-align:center;margin-bottom:4px}
    .hero h1{font-size:40px;font-weight:800;letter-spacing:-1px;line-height:1.1;background:linear-gradient(135deg,#fff 20%,#A78BFA);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:12px}
    .hero p{font-size:15px;color:var(--t2);line-height:1.7}
    .model-row{display:flex;gap:8px;background:var(--surface);border:1px solid var(--bds);border-radius:var(--r);padding:7px}
    .mbtn2{padding:11px 24px;border-radius:var(--rs);border:1px solid transparent;font-family:inherit;cursor:pointer;transition:all .2s;color:var(--t3);background:transparent;text-align:center}
    .mbtn2.on{background:var(--purd);border-color:var(--bdr);color:var(--purl)}
    .mbtn2:hover:not(.on){color:var(--t);background:rgba(255,255,255,.04)}
    .mbtn2 strong{display:block;font-size:14px;font-weight:600;margin-bottom:2px}
    .mbtn2 small{font-size:12px;opacity:.7}
    #view-upload{width:100%}
    .drop-zone{width:100%;border:2px dashed var(--bdr);border-radius:var(--r);background:var(--surface);padding:56px 32px;text-align:center;transition:all .25s}
    .drop-zone.over{border-color:var(--pur);background:var(--purd);box-shadow:0 0 36px rgba(124,58,237,.15)}
    .drop-icon{font-size:48px;display:block;margin-bottom:16px}
    .drop-zone h3{font-size:20px;font-weight:600;margin-bottom:8px}
    .drop-zone p{font-size:14px;color:var(--t3);margin-bottom:22px}
    .pick-btn{display:inline-block;padding:12px 30px;background:var(--pur);color:#fff;border-radius:var(--rs);font-weight:600;font-size:14px;border:none;cursor:pointer;font-family:inherit;box-shadow:0 4px 18px rgba(124,58,237,.4);transition:all .2s}
    .pick-btn:hover{background:#6D28D9;transform:translateY(-2px)}
    .formats{margin-top:20px;font-size:12px;color:var(--t3);display:flex;gap:7px;justify-content:center;flex-wrap:wrap}
    .formats span{background:var(--card);border:1px solid var(--bds);padding:3px 8px;border-radius:5px}
    #fileInput{display:none}
    #view-processing{display:none;width:100%;flex-direction:column;align-items:center;gap:24px;padding:56px 32px;background:var(--surface);border:1px solid var(--bds);border-radius:var(--r);text-align:center}
    .wave{display:flex;align-items:center;gap:5px;height:48px}
    .wb{width:5px;background:linear-gradient(to top,var(--pur),var(--purl));border-radius:3px;animation:wp 1.5s ease-in-out infinite}
    .wb:nth-child(1){animation-delay:.00s}.wb:nth-child(2){animation-delay:.12s}.wb:nth-child(3){animation-delay:.24s}
    .wb:nth-child(4){animation-delay:.36s}.wb:nth-child(5){animation-delay:.48s}.wb:nth-child(6){animation-delay:.36s}
    .wb:nth-child(7){animation-delay:.24s}.wb:nth-child(8){animation-delay:.12s}.wb:nth-child(9){animation-delay:.00s}
    @keyframes wp{0%,100%{height:8px;opacity:.4}50%{height:42px;opacity:1}}
    #proc-name{font-size:18px;font-weight:600}#proc-msg{font-size:14px;color:var(--t2)}
    .prog-wrap{width:100%;display:flex;flex-direction:column;align-items:center;gap:6px}
    .prog-outer{width:280px;height:6px;background:rgba(255,255,255,.08);border-radius:3px;overflow:hidden}
    .prog-fill{height:100%;background:linear-gradient(90deg,#7C3AED,#A78BFA);border-radius:3px;transition:width .6s ease;width:0%}
    .prog-pct{font-size:13px;font-weight:600;color:var(--purl)}
    .proc-note{font-size:13px;color:var(--t3);background:var(--card);border:1px solid var(--bds);padding:10px 20px;border-radius:9px}
    #view-results{display:none;width:100%;flex-direction:column;gap:16px}
    .res-hdr{display:flex;align-items:center;gap:11px}
    .chk{width:28px;height:28px;background:var(--grn);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;flex-shrink:0}
    .res-title{font-size:18px;font-weight:600}.res-file{font-size:13px;color:var(--t3);margin-top:2px}
    .presets{display:flex;align-items:center;gap:9px;flex-wrap:wrap;padding:13px 16px;background:var(--surface);border:1px solid var(--bds);border-radius:var(--r)}
    .pre-lbl{font-size:11px;font-weight:700;color:var(--t3);text-transform:uppercase;letter-spacing:1.2px;white-space:nowrap}
    .pre-btn{padding:6px 14px;border:1px solid var(--bds);border-radius:18px;background:transparent;color:var(--t2);font-size:12px;font-weight:500;cursor:pointer;font-family:inherit;transition:all .2s}
    .pre-btn:hover,.pre-btn.on{background:var(--purd);border-color:var(--bdr);color:var(--purl)}
    .stems-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(145px,1fr));gap:11px}
    .stem-card{background:var(--card);border:1px solid var(--bds);border-radius:var(--r);padding:13px 11px 11px;transition:all .2s}
    .stem-card:not(.muted):hover{background:var(--card2);border-color:var(--bdr);transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,.4)}
    .stem-card.muted{opacity:.3}.stem-card.soloed{border-color:rgba(245,158,11,.55)}
    .stem-card.cleaned{border-color:rgba(16,185,129,.3)}
    .stem-top{display:flex;justify-content:space-between;margin-bottom:7px}
    .cbtn{width:25px;height:25px;border-radius:6px;border:1px solid var(--bds);background:transparent;color:var(--t3);font-size:10px;font-weight:800;cursor:pointer;font-family:inherit;transition:all .15s;display:flex;align-items:center;justify-content:center}
    .cbtn:hover{border-color:var(--bdr);color:var(--t)}
    .mute-btn.on{background:rgba(239,68,68,.15);border-color:rgba(239,68,68,.4);color:#F87171}
    .solo-btn.on{background:rgba(245,158,11,.15);border-color:rgba(245,158,11,.4);color:#FCD34D}
    .stem-mid{text-align:center;padding:3px 0 7px}
    .stem-ico{font-size:28px;display:block;margin-bottom:6px}
    .stem-nm{font-size:13px;font-weight:600;text-transform:capitalize}
    .stem-clean-tag{font-size:9px;font-weight:700;color:var(--grn);letter-spacing:.5px;text-transform:uppercase;display:block;margin-top:2px}
    .vol-wrap{margin-top:9px}
    .vol-sl{width:100%;-webkit-appearance:none;appearance:none;height:4px;border-radius:2px;outline:none;cursor:pointer}
    .vol-sl::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;border-radius:50%;background:var(--pur);cursor:pointer;box-shadow:0 0 5px rgba(124,58,237,.6)}
    .vol-sl::-moz-range-thumb{width:13px;height:13px;border-radius:50%;background:var(--pur);cursor:pointer;border:none}
    .vol-lbl{text-align:center;font-size:11px;color:var(--t3);margin-top:5px}
    /* ── CLEAN & MASTER ── */
    .clean-box{background:var(--surface);border:1px solid var(--bds);border-radius:var(--r);padding:18px 20px;display:flex;flex-direction:column;gap:14px}
    .clean-title-row{display:flex;align-items:center;justify-content:space-between}
    .clean-title{font-size:15px;font-weight:600;display:flex;align-items:center;gap:8px}
    .clean-sub{font-size:12px;color:var(--t3);margin-top:3px}
    .int-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
    .int-lbl{font-size:11px;font-weight:700;color:var(--t3);text-transform:uppercase;letter-spacing:.8px;white-space:nowrap}
    .int-btn{padding:6px 16px;border:1px solid var(--bds);border-radius:18px;background:transparent;color:var(--t2);font-size:12px;font-weight:500;cursor:pointer;font-family:inherit;transition:all .2s}
    .int-btn.on{background:var(--purd);border-color:var(--bdr);color:var(--purl)}
    .int-btn:hover:not(.on){color:var(--t);border-color:var(--bdr)}
    .int-desc{font-size:12px;color:var(--t3);line-height:1.6;padding:2px 0}
    .clean-btn{width:100%;padding:12px;background:var(--card);border:1px solid var(--bds);border-radius:var(--rs);color:var(--t);font-family:inherit;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s}
    .clean-btn:hover:not(:disabled){background:var(--card2);border-color:rgba(16,185,129,.4);color:var(--grn)}
    .clean-btn:disabled{opacity:.5;cursor:not-allowed}
    .clean-prog{display:none;flex-direction:column;gap:10px}
    .clean-prog-msg{font-size:14px;color:var(--t2);font-weight:500}
    .clean-bar{height:4px;background:var(--bds);border-radius:2px;overflow:hidden}
    .clean-bar-fill{height:100%;width:100%;background:linear-gradient(90deg,#7C3AED,#10B981,#7C3AED);background-size:200%;animation:cleanScan 1.8s ease-in-out infinite}
    @keyframes cleanScan{0%{background-position:0%}100%{background-position:200%}}
    .clean-done{display:none;flex-direction:column;gap:4px}
    .clean-done-msg{font-size:14px;font-weight:600;color:var(--grn);display:flex;align-items:center;gap:8px}
    .clean-done-sub{font-size:12px;color:var(--t3)}
    /* ── A/B, Export ── */
    .ab-box{padding:16px 18px;background:var(--surface);border:1px solid var(--bds);border-radius:var(--r)}
    .box-lbl{font-size:11px;font-weight:700;color:var(--t3);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:13px}
    .ab-row{display:flex;gap:14px;flex-wrap:wrap}
    .ab-item{flex:1;min-width:190px}
    .ab-name{font-size:12px;color:var(--t2);margin-bottom:6px;font-weight:500}
    audio{width:100%;height:34px;border-radius:7px}
    .exp-bar{display:flex;align-items:center;gap:11px;padding:16px 18px;background:var(--surface);border:1px solid var(--bds);border-radius:var(--r)}
    .fmt-sel{padding:10px 13px;background:var(--card);border:1px solid var(--bds);border-radius:var(--rs);color:var(--t);font-family:inherit;font-size:14px;cursor:pointer;outline:none}
    .fmt-sel:focus{border-color:var(--bdr)}
    .exp-btn{flex:1;padding:11px 18px;background:var(--pur);color:#fff;border:none;border-radius:var(--rs);font-family:inherit;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s;box-shadow:0 4px 14px rgba(124,58,237,.35)}
    .exp-btn:hover:not(:disabled){background:#6D28D9;transform:translateY(-1px)}
    .exp-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
    .rst-btn{align-self:center;padding:11px 28px;background:transparent;border:1px solid var(--bds);border-radius:var(--rs);color:var(--t2);font-size:13px;font-weight:500;cursor:pointer;font-family:inherit;transition:all .2s}
    .rst-btn:hover{border-color:var(--bdr);color:var(--purl)}
    .err-box{width:100%;background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.3);border-radius:var(--r);padding:16px 20px;font-size:14px;color:#FCA5A5}
    .err-box strong{display:block;margin-bottom:5px;font-size:15px}
    /* ── AI REMIX ── */
    .ai-box{background:var(--surface);border:1px solid var(--bds);border-radius:var(--r);overflow:hidden}
    .ai-hdr{display:flex;align-items:center;justify-content:space-between;padding:16px 18px;cursor:pointer;user-select:none}
    .ai-hdr:hover{background:rgba(255,255,255,.02)}
    .ai-title{display:flex;align-items:center;gap:10px;font-size:15px;font-weight:600}
    .ai-badge{font-size:11px;font-weight:600;padding:3px 10px;border-radius:12px;border:1px solid transparent}
    .ai-badge.ready{background:rgba(16,185,129,.12);border-color:rgba(16,185,129,.3);color:#34D399}
    .ai-badge.loading{background:rgba(245,158,11,.1);border-color:rgba(245,158,11,.3);color:#FCD34D}
    .ai-badge.offline{background:rgba(255,255,255,.04);border-color:var(--bds);color:var(--t3)}
    .ai-chevron{background:transparent;border:none;color:var(--t3);font-size:13px;cursor:pointer;transition:transform .25s;padding:5px 8px}
    .ai-chevron.open{transform:rotate(180deg)}
    .ai-body{padding:0 18px 18px;flex-direction:column;gap:14px}
    .ai-mode-row{display:flex;gap:6px;background:var(--card);border:1px solid var(--bds);border-radius:var(--rs);padding:5px}
    .ai-mode-btn{flex:1;padding:9px 6px;border-radius:7px;border:1px solid transparent;background:transparent;color:var(--t3);font-family:inherit;font-size:12px;font-weight:500;cursor:pointer;transition:all .2s;text-align:center;line-height:1.3}
    .ai-mode-btn.on{background:var(--purd);border-color:var(--bdr);color:var(--purl)}
    .ai-mode-btn:hover:not(.on){color:var(--t);background:rgba(255,255,255,.04)}
    .ai-mode-btn strong{display:block;font-size:13px;font-weight:600;margin-bottom:1px}
    .ai-mode-desc{font-size:12px;color:var(--t3);text-align:center;line-height:1.5;padding:2px 4px}
    .ai-field{display:flex;flex-direction:column;gap:6px}
    .ai-lbl{font-size:11px;font-weight:700;color:var(--t2);text-transform:uppercase;letter-spacing:.6px}
    .ai-ta{width:100%;min-height:72px;padding:11px 13px;background:var(--card);border:1px solid var(--bds);border-radius:var(--rs);color:var(--t);font-family:inherit;font-size:13px;resize:vertical;outline:none;line-height:1.5}
    .ai-ta:focus{border-color:var(--bdr)}
    .ai-row2{display:flex;gap:11px}
    .ai-half{flex:1;display:flex;flex-direction:column;gap:6px}
    .ai-inp{width:100%;padding:10px 12px;background:var(--card);border:1px solid var(--bds);border-radius:var(--rs);color:var(--t);font-family:inherit;font-size:13px;outline:none}
    .ai-inp:focus{border-color:var(--bdr)}
    .ai-strength-labels{display:flex;justify-content:space-between;font-size:11px;color:var(--t3);margin-top:5px}
    .ai-strength-val{font-weight:700;color:var(--purl)}
    .ai-strength-note{font-size:11px;color:var(--t3);text-align:center;line-height:1.5;padding:2px 0}
    .ai-gen-btn{width:100%;padding:13px;background:linear-gradient(135deg,#7C3AED,#9D6FF7);color:#fff;border:none;border-radius:var(--rs);font-family:inherit;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s;box-shadow:0 4px 16px rgba(124,58,237,.4)}
    .ai-gen-btn:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 6px 22px rgba(124,58,237,.55)}
    .ai-gen-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
    .ai-prog{text-align:center;padding:18px 0;display:flex;flex-direction:column;align-items:center;gap:14px}
    .ai-prog-msg{font-size:15px;font-weight:600}.ai-prog-note{font-size:12px;color:var(--t3)}
    .ai-res-hdr{font-size:13px;font-weight:600;color:var(--grn);margin-bottom:9px}
    .ai-audio{width:100%;height:34px;border-radius:7px;margin-bottom:11px}
    .ai-res-btns{display:flex;gap:10px}
    .ai-dl{flex:1;display:flex;align-items:center;justify-content:center;padding:10px;background:var(--pur);color:#fff;border-radius:var(--rs);font-size:13px;font-weight:600;text-decoration:none;transition:all .2s}
    .ai-dl:hover{background:#6D28D9}
    .ai-again{flex:1;padding:10px;background:transparent;border:1px solid var(--bds);border-radius:var(--rs);color:var(--t2);font-size:13px;font-weight:500;cursor:pointer;font-family:inherit;transition:all .2s}
    .ai-again:hover{border-color:var(--bdr);color:var(--purl)}
    .ai-info{font-size:13px;color:var(--t3);line-height:1.7;padding:4px 0}
    .ai-info code{background:var(--card);padding:2px 7px;border-radius:4px;font-size:12px;color:var(--purl)}
    .ai-wait{display:flex;align-items:center;gap:12px;font-size:13px;color:var(--t2);padding:4px 0}
    .ai-dot{width:8px;height:8px;border-radius:50%;background:var(--ylw);animation:pulse 1.2s ease-in-out infinite;flex-shrink:0}
    @keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
    .ai-instr{display:flex;flex-direction:column;gap:8px}
    .ai-step{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--t2)}
    .ai-step-n{width:22px;height:22px;border-radius:50%;background:var(--purd);border:1px solid var(--bdr);display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:var(--purl);flex-shrink:0}
    /* ── steps ── */
    .steps{width:100%;display:flex;gap:14px}
    .step{flex:1;text-align:center;padding:20px 14px;background:var(--surface);border:1px solid var(--bds);border-radius:var(--r);transition:border-color .2s}
    .step:hover{border-color:var(--bdr)}
    .step-n{font-size:10px;font-weight:700;color:var(--purl);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:9px}
    .step-i{font-size:26px;display:block;margin-bottom:9px}
    .step h4{font-size:13px;font-weight:600;margin-bottom:5px}
    .step p{font-size:12px;color:var(--t3);line-height:1.6}
    /* ── VOCAL SPLIT ── */
    .sv-sec{margin-top:8px;padding-top:8px;border-top:1px solid rgba(16,185,129,.2)}
    .sv-btn{width:100%;padding:7px 6px;border:1px solid rgba(16,185,129,.35);border-radius:var(--rs);background:rgba(16,185,129,.08);color:#34D399;font-family:inherit;font-size:11px;font-weight:600;cursor:pointer;transition:all .2s;display:flex;align-items:center;justify-content:center;gap:5px}
    .sv-btn:hover:not(:disabled){background:rgba(16,185,129,.15);border-color:rgba(16,185,129,.55)}
    .sv-btn:disabled{opacity:.5;cursor:not-allowed}
    .sv-msg{font-size:10px;color:var(--t3);text-align:center;margin-top:5px;min-height:13px;line-height:1.4}
    /* ── STEM AI REPLACEMENT ── */
    .ai-rep-sec{margin-top:9px;padding-top:9px;border-top:1px solid var(--bds)}
    .ai-rep-toggle{width:100%;padding:6px 8px;border:1px solid var(--bds);border-radius:var(--rs);background:transparent;color:var(--t3);font-family:inherit;font-size:11px;font-weight:600;cursor:pointer;transition:all .2s;text-align:center;display:flex;align-items:center;justify-content:center;gap:5px}
    .ai-rep-toggle:hover{border-color:var(--bdr);color:var(--purl);background:var(--purd)}
    .ai-rep-toggle.on{border-color:var(--bdr);color:var(--purl);background:var(--purd)}
    .ai-rep-panel{margin-top:8px;display:none;flex-direction:column;gap:6px}
    .ai-rep-inp{width:100%;padding:6px 8px;background:var(--card2);border:1px solid var(--bds);border-radius:6px;color:var(--t);font-family:inherit;font-size:11px;outline:none;resize:none;line-height:1.4}
    .ai-rep-inp:focus{border-color:var(--bdr)}
    .ai-rep-gen{width:100%;padding:7px;background:linear-gradient(135deg,#7C3AED,#9D6FF7);color:#fff;border:none;border-radius:6px;font-family:inherit;font-size:11px;font-weight:600;cursor:pointer;transition:all .2s;box-shadow:0 2px 8px rgba(124,58,237,.35)}
    .ai-rep-gen:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 4px 12px rgba(124,58,237,.5)}
    .ai-rep-gen:disabled{opacity:.5;cursor:not-allowed;transform:none}
    .ai-rep-msg{font-size:10px;color:var(--t3);text-align:center;min-height:14px}
    .ai-rep-revert{width:100%;padding:5px;border:1px solid rgba(239,68,68,.3);border-radius:6px;background:transparent;color:#F87171;font-family:inherit;font-size:10px;font-weight:500;cursor:pointer;transition:all .2s}
    .ai-rep-revert:hover{background:rgba(239,68,68,.1)}
    .ai-badge-rep{font-size:9px;font-weight:700;color:#A78BFA;letter-spacing:.8px;text-transform:uppercase;display:none;margin-top:2px}
    .stem-card.ai-replaced{border-color:rgba(124,58,237,.5)!important;box-shadow:0 0 12px rgba(124,58,237,.15)}
    /* ── KARAOKE EDITOR ── */
    #view-karaoke{display:none;width:100%;flex-direction:column;gap:16px}
    .k-box{background:var(--surface);border:1px solid var(--bds);border-radius:var(--r);padding:18px 20px;display:flex;flex-direction:column;gap:12px}
    .k-stitle{font-size:11px;font-weight:700;color:var(--t2);text-transform:uppercase;letter-spacing:.8px}
    #waveform{background:var(--card);border-radius:8px;overflow:hidden;min-height:80px;cursor:crosshair}
    #waveform wave{display:block!important}
    #wf-timeline{margin-top:3px}
    .wf-ctrls{display:flex;align-items:center;gap:8px;margin-top:8px;flex-wrap:wrap}
    .wf-btn{width:32px;height:32px;border-radius:8px;border:1px solid var(--bds);background:transparent;color:var(--t);font-size:13px;cursor:pointer;font-family:inherit;transition:all .2s;display:flex;align-items:center;justify-content:center}
    .wf-btn:hover{background:var(--purd);border-color:var(--bdr)}
    .wf-time{font-size:13px;color:var(--t2);flex:1;font-variant-numeric:tabular-nums}
    .k-label-row{display:flex;gap:8px;flex-wrap:wrap}
    .k-label-btn{display:flex;align-items:center;gap:7px;padding:8px 16px;border:1px solid var(--bds);border-radius:20px;background:transparent;color:var(--t2);font-size:13px;font-weight:500;cursor:pointer;font-family:inherit;transition:all .2s}
    .k-label-btn.on{background:var(--card2);border-color:var(--bdr);color:var(--t)}
    .k-label-btn:hover:not(.on){border-color:var(--bdr);color:var(--t)}
    .k-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;display:inline-block}
    .k-hint{font-size:12px;color:var(--t3);line-height:1.5}
    .k-regions-empty{font-size:13px;color:var(--t3);padding:8px 0}
    .k-region-item{display:flex;align-items:center;gap:10px;padding:9px 12px;background:var(--card);border-radius:var(--rs);margin-bottom:6px}
    .k-region-swatch{width:12px;height:30px;border-radius:4px;flex-shrink:0}
    .k-region-times{font-size:13px;color:var(--t);font-variant-numeric:tabular-nums;flex:1}
    .k-region-lbl{font-size:12px;color:var(--t2);font-weight:500;min-width:72px}
    .k-region-del{width:24px;height:24px;border-radius:5px;border:none;background:rgba(239,68,68,.1);color:#F87171;cursor:pointer;font-size:16px;line-height:1;transition:all .15s;display:flex;align-items:center;justify-content:center}
    .k-region-del:hover{background:rgba(239,68,68,.25)}
    .k-clear{align-self:flex-start;padding:6px 14px;border:1px solid rgba(239,68,68,.25);border-radius:18px;background:transparent;color:#F87171;font-size:12px;font-weight:500;cursor:pointer;font-family:inherit;transition:all .2s}
    .k-clear:hover{background:rgba(239,68,68,.1)}
    .k-silence-row{display:flex;gap:8px;flex-wrap:wrap}
    .k-sil-btn{padding:8px 20px;border:1px solid var(--bds);border-radius:18px;background:transparent;color:var(--t2);font-size:13px;font-weight:500;cursor:pointer;font-family:inherit;transition:all .2s}
    .k-sil-btn.on{background:var(--purd);border-color:var(--bdr);color:var(--purl)}
    .k-sil-btn:hover:not(.on){color:var(--t);border-color:var(--bdr)}
    .k-fmt-row{display:flex;gap:11px}
    .k-exp-btn{flex:1;padding:11px 18px;background:linear-gradient(135deg,#7C3AED,#9D6FF7);color:#fff;border:none;border-radius:var(--rs);font-family:inherit;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s;box-shadow:0 4px 14px rgba(124,58,237,.35)}
    .k-exp-btn:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 6px 20px rgba(124,58,237,.5)}
    .k-exp-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
    .k-result-box{background:var(--surface);border:1px solid rgba(16,185,129,.3);border-radius:var(--r);padding:18px 20px;display:none;flex-direction:column;gap:12px}
    .k-result-hdr{font-size:14px;font-weight:600;color:var(--grn)}
    .k-result-box audio{width:100%;height:34px;border-radius:7px}
    .k-dl{display:flex;align-items:center;justify-content:center;padding:10px;background:var(--pur);color:#fff;border-radius:var(--rs);font-size:13px;font-weight:600;text-decoration:none;transition:all .2s}
    .k-dl:hover{background:#6D28D9}
    .k-sub{font-size:13px;color:var(--t3)}
    .wf-right{display:flex;align-items:center;gap:8px;margin-left:auto;flex-wrap:wrap}
    .wf-mode-btn{padding:5px 12px;border:1px solid var(--bds);border-radius:18px;background:transparent;color:var(--t2);font-size:12px;font-weight:500;cursor:pointer;font-family:inherit;transition:all .2s;white-space:nowrap}
    .wf-mode-btn.on{background:var(--purd);border-color:var(--bdr);color:var(--purl)}
    .wf-mode-btn:hover:not(.on){color:var(--t);border-color:var(--bdr)}
    .wf-zoom-lbl{font-size:12px;color:var(--t3);white-space:nowrap}
    .wf-vdiv{width:1px;height:18px;background:var(--bds);flex-shrink:0}
    #wfZoom{width:80px;accent-color:#7C3AED;cursor:pointer}
    .k-region-sel{padding:4px 8px;background:var(--card2);border:1px solid var(--bds);border-radius:6px;color:var(--t);font-family:inherit;font-size:12px;cursor:pointer;outline:none;min-width:84px}
    .k-region-sel:focus{border-color:var(--bdr)}

    /* ── YOUTUBE DOWNLOADER ── */
    .yt-box{background:var(--surface);border:1px solid var(--bds);border-radius:var(--r);padding:18px 20px;display:flex;flex-direction:column;gap:12px;margin-bottom:0}
    .yt-head{font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px}
    .yt-head-note{font-size:12px;color:var(--t3);font-weight:400}
    .yt-row{display:flex;gap:10px}
    .yt-inp{flex:1;padding:10px 14px;background:var(--card);border:1px solid var(--bds);border-radius:var(--rs);color:var(--t);font-family:inherit;font-size:13px;outline:none}
    .yt-inp:focus{border-color:var(--bdr)}
    .yt-cvt-btn{padding:10px 20px;background:#CC0000;color:#fff;border:none;border-radius:var(--rs);font-family:inherit;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;white-space:nowrap}
    .yt-cvt-btn:hover:not(:disabled){background:#AA0000;transform:translateY(-1px)}
    .yt-cvt-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
    .yt-prog-wrap{display:flex;align-items:center;gap:10px}
    .yt-prog-outer{flex:1;height:5px;background:rgba(255,255,255,.08);border-radius:3px;overflow:hidden}
    .yt-prog-fill{height:100%;background:linear-gradient(90deg,#CC0000,#FF6B6B);border-radius:3px;transition:width .4s ease;width:0%}
    .yt-prog-pct{font-size:12px;font-weight:600;color:var(--t2);min-width:36px;text-align:right}
    .yt-msg{font-size:12px;color:var(--t3)}
    .yt-result{display:flex;flex-direction:column;gap:10px}
    .yt-result-title{font-size:13px;color:var(--t);font-weight:500;display:-webkit-box;-webkit-line-clamp:1;-webkit-box-orient:vertical;overflow:hidden}
    .yt-result-btns{display:flex;gap:9px}
    .yt-dl-btn{padding:9px 16px;background:var(--card);border:1px solid var(--bds);border-radius:var(--rs);color:var(--t);font-size:12px;font-weight:500;text-decoration:none;transition:all .2s;white-space:nowrap;display:flex;align-items:center;gap:5px}
    .yt-dl-btn:hover{border-color:var(--bdr);color:var(--purl)}
    .yt-sep-btn{flex:1;padding:9px 16px;background:linear-gradient(135deg,#7C3AED,#9D6FF7);color:#fff;border:none;border-radius:var(--rs);font-family:inherit;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;box-shadow:0 3px 10px rgba(124,58,237,.35)}
    .yt-sep-btn:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 5px 14px rgba(124,58,237,.5)}
    .yt-sep-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
    .yt-browser-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 0 2px}
    .yt-browser-lbl{font-size:12px;color:var(--t2);font-weight:500;white-space:nowrap}
    .yt-browser-sel{padding:5px 10px;background:var(--card);border:1px solid var(--bds);border-radius:var(--rs);color:var(--t);font-family:inherit;font-size:12px;cursor:pointer;outline:none}
    .yt-browser-sel:focus{border-color:var(--bdr)}
    .yt-browser-hint{font-size:11px;color:var(--t3);flex:1}
    .yt-music-hint{display:none;font-size:12px;color:#f0a500;background:rgba(240,165,0,.1);border:1px solid rgba(240,165,0,.3);border-radius:var(--rs);padding:8px 12px;line-height:1.5}
    .yt-login-btn{padding:5px 12px;background:var(--card);border:1px solid var(--bds);border-radius:var(--rs);color:var(--t2);font-family:inherit;font-size:11px;cursor:pointer;white-space:nowrap;transition:all .2s}
    .yt-login-btn:hover{border-color:var(--bdr);color:var(--t)}
    .yt-divider{display:flex;align-items:center;gap:12px;color:var(--t3);font-size:12px;margin:4px 0}
    .yt-divider::before,.yt-divider::after{content:'';flex:1;height:1px;background:var(--bds)}
    /* ── PROJECTS PANEL ── */
    .pj-section{background:var(--surface);border:1px solid var(--bds);border-radius:var(--r);overflow:hidden}
    .pj-hdr{display:flex;align-items:center;justify-content:space-between;padding:11px 16px;cursor:pointer;user-select:none}
    .pj-hdr:hover{background:rgba(255,255,255,.02)}
    .pj-hdr-left{font-size:12px;font-weight:600;color:var(--t2);display:flex;align-items:center;gap:6px}
    .pj-hdr-right{display:flex;align-items:center;gap:8px}
    .pj-save-btn{padding:4px 12px;border:1px solid var(--bds);border-radius:var(--rs);background:transparent;color:var(--t2);font-family:inherit;font-size:11px;font-weight:500;cursor:pointer;transition:all .2s;display:none}
    .pj-save-btn:hover{border-color:var(--bdr);color:var(--purl)}
    .pj-chev{font-size:9px;color:var(--t3);transition:transform .2s}
    .pj-chev.open{transform:rotate(180deg)}
    .pj-body{padding:0 12px 12px;display:block}
    .pj-body.closed{display:none}
    .pj-scroll{display:flex;gap:10px;overflow-x:auto;padding-bottom:4px;scrollbar-width:thin;scrollbar-color:var(--bds) transparent}
    .pj-card{flex:0 0 auto;background:var(--card);border:1px solid var(--bds);border-radius:var(--rs);padding:0;cursor:pointer;transition:all .2s;min-width:170px;max-width:200px;overflow:hidden;position:relative}
    .pj-card:hover{border-color:var(--bdr);transform:translateY(-1px)}
    .pj-card-top{height:3px}
    .pj-card-body{padding:9px 10px}
    .pj-card-name{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:4px}
    .pj-card-meta{font-size:10px;color:var(--t3);display:flex;gap:6px;flex-wrap:wrap}
    .pj-del{position:absolute;top:5px;right:5px;width:16px;height:16px;border-radius:3px;border:none;background:transparent;color:var(--t3);cursor:pointer;font-size:11px;display:none;align-items:center;justify-content:center;line-height:1;padding:0}
    .pj-card:hover .pj-del{display:flex}
    .pj-del:hover{background:rgba(239,68,68,.2);color:#F87171}
    .pj-hint{font-size:11px;color:var(--t3);padding:2px 2px 0;font-style:italic}
    .pj-empty{font-size:12px;color:var(--t3);padding:6px 2px}
\
    /* ── AUTO-CLASSIFY ── */
    .k-auto-btn{width:100%;padding:9px;border:1px dashed rgba(124,58,237,.5);border-radius:var(--rs);background:var(--purd);color:var(--purl);font-family:inherit;font-size:12px;font-weight:600;cursor:pointer;transition:all .2s}
    .k-auto-btn:hover:not(:disabled){background:rgba(124,58,237,.2);border-style:solid}
    .k-auto-btn:disabled{opacity:.5;cursor:not-allowed}
    .k-auto-status{display:none;flex-direction:column;gap:6px;margin-top:2px}
    .k-auto-msg{font-size:11px;color:var(--t2)}
    .k-auto-bar{height:4px;background:rgba(255,255,255,.08);border-radius:2px;overflow:hidden}
    .k-auto-fill{height:100%;background:linear-gradient(90deg,#7C3AED,#A78BFA);border-radius:2px;transition:width .4s;width:0%}
    /* ── SETTINGS PANEL ── */
    .cfg-section{background:var(--surface);border:1px solid var(--bds);border-radius:var(--r);overflow:hidden;margin-bottom:0}
    .cfg-hdr{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;cursor:pointer}
    .cfg-hdr:hover{background:rgba(255,255,255,.02)}
    .cfg-hdr-left{font-size:12px;font-weight:600;color:var(--t2);display:flex;align-items:center;gap:6px}
    .cfg-body{padding:14px 16px;display:none;flex-direction:column;gap:12px}
    .cfg-body.open{display:flex}
    .cfg-row{display:flex;flex-direction:column;gap:5px}
    .cfg-lbl{font-size:11px;font-weight:600;color:var(--t2)}
    .cfg-sub{font-size:10px;color:var(--t3);line-height:1.4}
    .cfg-inp{padding:7px 10px;background:var(--card);border:1px solid var(--bds);border-radius:var(--rs);color:var(--t);font-family:monospace;font-size:12px;outline:none;width:100%;box-sizing:border-box}
    .cfg-inp:focus{border-color:var(--bdr)}
    .cfg-save{padding:7px 18px;background:var(--pur);color:#fff;border:none;border-radius:var(--rs);font-family:inherit;font-size:12px;font-weight:600;cursor:pointer;align-self:flex-start;transition:all .2s}
    .cfg-save:hover{background:#6D28D9}
    .cfg-status{font-size:11px;color:var(--grn)}
    .cfg-badge{font-size:10px;font-weight:600;padding:2px 7px;border-radius:4px;margin-left:6px}
    .cfg-badge.set{background:rgba(16,185,129,.15);color:#34D399}
    .cfg-badge.unset{background:rgba(255,255,255,.06);color:var(--t3)}
    footer{margin-top:auto;padding:22px;font-size:12px;color:var(--t3);text-align:center;border-top:1px solid var(--bds);width:100%}
  </style>
</head>
<body>
<header>
  <div class="logo"><div class="logo-i">&#127925;</div><span class="logo-t">Stem<span>Split</span></span></div>
  <span class="badge">Demucs + AceStep AI</span>
</header>

<main>
  <div class="hero">
    <h1>Separate, Clean &amp; Remix</h1>
    <p>Isolate stems &middot; Remove artifacts &middot; Remaster &middot; AI Remix<br>Everything runs locally on your machine.</p>
  </div>

  <div class="pj-section" id="pjSection">
    <div class="pj-hdr" id="pjHdr">
      <div class="pj-hdr-left">&#128193; Recent Projects</div>
      <div class="pj-hdr-right" onclick="event.stopPropagation()">
        <button class="pj-save-btn" id="pjSaveBtn">&#128190; Save</button>
        <span class="pj-chev open" id="pjChev">&#9660;</span>
      </div>
    </div>
    <div class="pj-body" id="pjBody">
      <div class="pj-scroll" id="pjScroll">
        <div class="pj-empty">Loading...</div>
      </div>
      <div class="pj-hint">Separate a track to create a new project automatically</div>
    </div>
  </div>

  <div class="cfg-section" id="cfgSection">
    <div class="cfg-hdr" id="cfgHdr">
      <div class="cfg-hdr-left">
        &#9881; AI Settings
        <span class="cfg-badge unset" id="cfgGeniusBadge">Genius: not set</span>
        <span class="cfg-badge unset" id="cfgHfBadge">HuggingFace: not set</span>
      </div>
      <span class="pj-chev" id="cfgChev">&#9660;</span>
    </div>
    <div class="cfg-body" id="cfgBody">
      <div class="cfg-row">
        <div class="cfg-lbl">Genius API Key <span style="font-weight:400;color:var(--t3)">(for lyrics-based singer detection)</span></div>
        <div class="cfg-sub">Free at <strong>genius.com/api-clients</strong> &rarr; New API Client &rarr; copy the Client Access Token</div>
        <input class="cfg-inp" type="password" id="cfgGenius" placeholder="Paste Genius Client Access Token...">
      </div>
      <div class="cfg-row">
        <div class="cfg-lbl">HuggingFace Token <span style="font-weight:400;color:var(--t3)">(for AI diarization via pyannote)</span></div>
        <div class="cfg-sub">Free at <strong>hf.co/settings/tokens</strong>. Also accept the model license at <strong>hf.co/pyannote/speaker-diarization-3.1</strong></div>
        <input class="cfg-inp" type="password" id="cfgHf" placeholder="Paste HuggingFace token...">
      </div>
      <div style="display:flex;align-items:center;gap:12px">
        <button class="cfg-save" id="cfgSaveBtn">Save Keys</button>
        <span class="cfg-status" id="cfgStatus"></span>
      </div>
      <div class="cfg-sub" style="border-top:1px solid var(--bds);padding-top:10px;margin-top:2px">
        <strong>Detection priority:</strong> Genius lyrics (most accurate for known songs) &rarr; pyannote AI diarization &rarr; voice fingerprinting (built-in fallback). Keys are stored locally in <code>stemsplit_config.json</code> and never sent anywhere.
      </div>
    </div>
  </div>

  <div class="model-row">
    <button class="mbtn2 on" data-model="htdemucs" id="m4"><strong>4 Stems</strong><small>Vocals &middot; Drums &middot; Bass &middot; Other</small></button>
    <button class="mbtn2" data-model="htdemucs_6s" id="m6"><strong>6 Stems</strong><small>+ Guitar &middot; Piano</small></button>
    <button class="mbtn2" id="mKar"><strong>&#127908; Karaoke</strong><small>Vocals + Instrumental</small></button>
  </div>

  <div id="view-upload">
    <div class="yt-box">
      <div class="yt-head">&#9654;&#65039; YouTube / YouTube Music <span class="yt-head-note">personal use only</span></div>
      <div class="yt-row">
        <input class="yt-inp" type="text" id="ytUrl" placeholder="Paste a YouTube or YouTube Music URL...">
        <button class="yt-cvt-btn" id="ytConvert">Convert</button>
      </div>
      <div id="ytMusicHint" class="yt-music-hint">
        &#127357; YouTube Music URL detected. To download from your library: make sure you are logged in to YouTube in the browser you select below, then click Convert.
        <br><button class="yt-login-btn" id="ytOpenBrowser" style="margin-top:6px">Open YouTube Music in browser to log in</button>
      </div>
      <div class="yt-browser-row">
        <span class="yt-browser-lbl">&#127850; Browser cookies</span>
        <select id="ytBrowser" class="yt-browser-sel">
          <option value="">None (public videos only)</option>
          <option value="chrome">Chrome</option>
          <option value="firefox">Firefox</option>
          <option value="edge">Edge</option>
          <option value="opera">Opera</option>
        </select>
        <span class="yt-browser-hint">Required for YouTube Music &amp; private videos &#8212; browser must be logged in to YouTube</span>
      </div>
      <div id="ytStatus" style="display:none;flex-direction:column;gap:8px">
        <div class="yt-prog-wrap">
          <div class="yt-prog-outer"><div class="yt-prog-fill" id="ytProgFill"></div></div>
          <div class="yt-prog-pct" id="ytProgPct">0%</div>
        </div>
        <div class="yt-msg" id="ytMsg">Starting...</div>
      </div>
      <div class="yt-result" id="ytResult" style="display:none">
        <div class="yt-result-title" id="ytResultTitle"></div>
        <div class="yt-result-btns">
          <a class="yt-dl-btn" id="ytDlBtn" download>&#11015; Download MP3</a>
          <button class="yt-sep-btn" id="ytSepBtn">&#127925; Separate &amp; Remix</button>
        </div>
      </div>
    </div>
    <div class="yt-divider"><span>or upload your own file</span></div>
    <input type="file" id="fileInput" accept=".mp3,.wav,.flac,.m4a,.ogg,.aiff">
    <div class="drop-zone" id="dropZone">
      <span class="drop-icon">&#127925;</span>
      <h3>Drop your audio file here</h3>
      <p>or click the button to browse your computer</p>
      <button class="pick-btn" id="pickBtn">Select File</button>
      <div class="formats"><span>MP3</span><span>WAV</span><span>FLAC</span><span>M4A</span><span>OGG</span><span>AIFF</span></div>
    </div>
  </div>

  <div id="view-processing">
    <div class="wave"><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div></div>
    <div id="proc-name">Processing...</div><div id="proc-msg">Starting AI separation...</div>
    <div id="progWrap" style="display:none;flex-direction:column;align-items:center;gap:6px;width:100%">
      <div style="width:280px;height:6px;background:rgba(255,255,255,.08);border-radius:3px;overflow:hidden"><div id="progFill" style="height:100%;background:linear-gradient(90deg,#7C3AED,#A78BFA);border-radius:3px;transition:width .6s ease;width:0%"></div></div>
      <div id="progPct" style="font-size:13px;font-weight:600;color:#A78BFA">0%</div>
    </div>
    <div class="proc-note">&#9201; Typically 2&ndash;5 minutes depending on track length</div>
  </div>

  <div id="view-results">
    <div class="res-hdr">
      <div class="chk">&#10003;</div>
      <div><div class="res-title">Separation complete!</div><div class="res-file" id="resFile"></div></div>
    </div>

    <div class="presets">
      <span class="pre-lbl">Presets</span>
      <button class="pre-btn" data-preset="karaoke">&#127908; Karaoke</button>
      <button class="pre-btn" data-preset="acapella">&#127925; Acapella</button>
      <button class="pre-btn" data-preset="instrumental">&#127932; Instrumental</button>
      <button class="pre-btn" data-preset="drumsandbass">&#129345; Drums+Bass</button>
      <button class="pre-btn" data-preset="reset">&#8635; Full Mix</button>
    </div>

    <div class="stems-grid" id="stemsGrid"></div>

    <!-- ── CLEAN & MASTER ── -->
    <div class="clean-box">
      <div class="clean-title-row">
        <div>
          <div class="clean-title">&#127911; Clean &amp; Master</div>
          <div class="clean-sub">Reduce separation artifacts, apply stem-specific EQ &amp; compression, normalize loudness</div>
        </div>
      </div>
      <div id="cleanIdle" style="display:flex;flex-direction:column;gap:12px">
        <div class="int-row">
          <span class="int-lbl">Intensity</span>
          <button class="int-btn" data-val="light" id="intLight">Light</button>
          <button class="int-btn on" data-val="medium" id="intMedium">Medium</button>
          <button class="int-btn" data-val="heavy" id="intHeavy">Heavy</button>
        </div>
        <div class="int-desc" id="intDesc">Moderate noise reduction, stem-specific EQ, 3.5:1 compression, loudness normalization. Good for most tracks.</div>
        <button class="clean-btn" id="cleanBtn">&#127911; Clean &amp; Master All Stems</button>
      </div>
      <div class="clean-prog" id="cleanProg">
        <div class="clean-prog-msg" id="cleanMsg">Starting cleanup...</div>
        <div class="clean-bar"><div class="clean-bar-fill"></div></div>
      </div>
      <div class="clean-done" id="cleanDone">
        <div class="clean-done-msg">&#10003; All stems cleaned &amp; remastered</div>
        <div class="clean-done-sub">Stems updated with noise reduction, EQ, compression and loudness normalization. Export your mix to hear the difference.</div>
      </div>
    </div>
    <!-- ── /CLEAN & MASTER ── -->

    <div class="ab-box" id="abBox" style="display:none">
      <div class="box-lbl">&#8644; A/B Compare</div>
      <div class="ab-row">
        <div class="ab-item"><div class="ab-name">&#127925; Original</div><audio id="origAudio" controls preload="none"></audio></div>
        <div class="ab-item"><div class="ab-name">&#127381; Your Mix</div><audio id="mixAudio" controls preload="none"></audio></div>
      </div>
    </div>

    <div class="exp-bar">
      <select class="fmt-sel" id="fmtSel"><option value="mp3">MP3</option><option value="wav">WAV</option><option value="flac">FLAC</option></select>
      <button class="exp-btn" id="expBtn">&#11015; Export Mix</button>
    </div>

    <!-- ── AI REMIX ── -->
    <div class="ai-box">
      <div class="ai-hdr" id="aiHdr">
        <div class="ai-title"><span>&#10024;</span> AI Remix <span class="ai-badge offline" id="aiBadge">Checking...</span></div>
        <button class="ai-chevron" id="aiChevron">&#9660;</button>
      </div>
      <div class="ai-body" id="aiBody" style="display:none">
        <div id="aiOffline" style="display:none">
          <p class="ai-info">AceStep worker is not running. To enable AI Remix:</p>
          <div class="ai-instr">
            <div class="ai-step"><div class="ai-step-n">1</div>Run <code>setup_acestep.bat</code> once to install AceStep</div>
            <div class="ai-step"><div class="ai-step-n">2</div>Use <code>start_all.bat</code> instead of <code>start.bat</code></div>
            <div class="ai-step"><div class="ai-step-n">3</div>First run downloads ~5GB model weights automatically</div>
          </div>
        </div>
        <div id="aiLoading" style="display:none"><div class="ai-wait"><div class="ai-dot"></div>AI models loading into GPU memory, please wait...</div></div>
        <div id="aiReady" style="display:none;flex-direction:column;gap:14px">
          <div class="ai-mode-row">
            <button class="ai-mode-btn on" id="aiModeGenerate"><strong>&#10024; Generate Fresh</strong>New track from description</button>
            <button class="ai-mode-btn" id="aiModeTransfer"><strong>&#128260; Style Transfer</strong>Remix your actual track</button>
            <button class="ai-mode-btn" id="aiModeReference"><strong>&#127911; Reference Style</strong>Your track's vibe, new sound</button>
          </div>
          <div class="ai-mode-desc" id="aiModeDesc">Generate entirely new music from your text description</div>
          <div class="ai-field">
            <label class="ai-lbl">Style Description</label>
            <textarea class="ai-ta" id="aiPrompt" placeholder="e.g. lo-fi hip hop, soft piano, vinyl crackle, relaxing&#10;e.g. energetic rock, electric guitar, powerful drums"></textarea>
          </div>
          <div id="aiFreshCtrl" style="flex-direction:column;gap:11px">
            <div class="ai-row2">
              <div class="ai-half"><label class="ai-lbl">Duration (sec)</label><input class="ai-inp" type="number" id="aiDur" value="30" min="5" max="240"></div>
              <div class="ai-half"><label class="ai-lbl">BPM (optional)</label><input class="ai-inp" type="number" id="aiBpm" placeholder="auto" min="40" max="240"></div>
            </div>
          </div>
          <div id="aiTransferCtrl" style="display:none;flex-direction:column;gap:8px">
            <div class="ai-field">
              <label class="ai-lbl">Faithfulness to Original <span class="ai-strength-val" id="aiStrengthVal">0.4</span></label>
              <input type="range" class="vol-sl" id="aiStrength" min="1" max="9" value="4">
              <div class="ai-strength-labels"><span>More Creative</span><span>Balanced</span><span>More Faithful</span></div>
            </div>
            <div class="ai-strength-note">Lower = bolder style change. Higher = preserves original melody &amp; structure. 0.3&ndash;0.5 recommended.</div>
          </div>
          <div id="aiReferenceCtrl" style="display:none;flex-direction:column;gap:11px">
            <div class="ai-row2">
              <div class="ai-half"><label class="ai-lbl">Duration (sec)</label><input class="ai-inp" type="number" id="aiDur2" value="30" min="5" max="240"></div>
              <div class="ai-half"><label class="ai-lbl">BPM (optional)</label><input class="ai-inp" type="number" id="aiBpm2" placeholder="auto" min="40" max="240"></div>
            </div>
          </div>
          <button class="ai-gen-btn" id="aiGenBtn">&#10024; Generate</button>
        </div>
        <div id="aiGenerating" style="display:none">
          <div class="ai-prog">
            <div class="wave" style="height:36px"><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div></div>
            <div class="ai-prog-msg" id="aiProgMsg">Starting...</div>
            <div class="ai-prog-note">AI remix takes 1&ndash;3 minutes on GPU</div>
          </div>
        </div>
        <div id="aiResult" style="display:none;flex-direction:column">
          <div class="ai-res-hdr" id="aiResHdr">&#10003; AI Remix ready</div>
          <audio class="ai-audio" id="aiAudio" controls preload="auto"></audio>
          <div class="ai-res-btns">
            <a class="ai-dl" id="aiDlBtn" download="ai_remix.mp3">&#11015; Download</a>
            <button class="ai-again" id="aiAgainBtn">&#8635; Generate Another</button>
          </div>
        </div>
      </div>
    </div>

    <button class="rst-btn" id="rstBtn">&#8617; Process Another Track</button>
  </div>


  <div id="view-karaoke">
    <div class="res-hdr">
      <div class="chk">&#127908;</div>
      <div><div class="res-title">Karaoke Editor</div><div class="res-file" id="kFile"></div></div>
    </div>

    <div class="k-box">
      <div class="k-stitle">Vocal Track &mdash; Click &amp; Drag to Mark Regions</div>
      <div id="waveform"></div>
      <div id="wf-timeline"></div>
      <div class="wf-ctrls">
        <button class="wf-btn" id="wfPlay" title="Play/Pause">&#9654;</button>
        <button class="wf-btn" id="wfStop" title="Stop">&#9209;</button>
        <span class="wf-time" id="wfTime">0:00 / 0:00</span>
        <div class="wf-right">
          <button class="wf-mode-btn on" id="wfDrawMode" title="Drag on waveform to create regions">&#9998; Draw</button>
          <button class="wf-mode-btn" id="wfSeekMode" title="Click waveform to seek/play">&#8594; Seek</button>
          <span class="wf-vdiv"></span>
          <span class="wf-zoom-lbl">Zoom</span>
          <input type="range" id="wfZoom" min="0" max="600" value="0" step="20" title="Zoom in for precise marking">
          <button class="wf-btn" id="wfZoomReset" title="Reset zoom">&#8634;</button>
        </div>
      </div>
    </div>

    <div class="k-box">
      <div class="k-stitle">Label New Regions As</div>
      <div class="k-label-row">
        <button class="k-label-btn on" data-klabel="Singer A" id="kLblA"><span class="k-dot" style="background:#7C3AED"></span>Singer A</button>
        <button class="k-label-btn" data-klabel="Singer B" id="kLblB"><span class="k-dot" style="background:#10B981"></span>Singer B</button>
        <button class="k-label-btn" data-klabel="Both" id="kLblBoth"><span class="k-dot" style="background:#F59E0B"></span>Both</button>
      </div>
      <div class="k-hint">&#8592; Pick a label above, then drag on the waveform to mark that segment. Regions can be dragged and resized after creation.</div>
    </div>

    <div class="k-box" id="kRegionsBox">
      <div class="k-stitle">Marked Regions</div>
      <div id="kRegionsList"><div class="k-regions-empty">No regions marked yet. Drag on the waveform above to add one.</div></div>
\
      <div style="display:flex;flex-direction:column;gap:6px;margin-bottom:2px">
        <label style="font-size:11px;font-weight:600;color:var(--t2)">Song title for lyrics search <span style="font-weight:400;color:var(--t3)">(Artist &mdash; Song Name)</span></label>
        <input type="text" id="kSongTitle" class="ai-inp" placeholder="e.g. Beyonce - Crazy in Love" style="font-size:12px">
      </div>
      <div style="display:flex;gap:8px">
        <button class="k-auto-btn" id="kAutoBtn" style="flex:1">&#129302; Auto-detect Singers</button>
        <button class="k-auto-btn" id="kTestBtn" style="flex:0 0 auto;padding:9px 14px;font-size:11px;opacity:.8" title="Test Genius lookup for this song">&#128270; Test</button>
      </div>
      <div class="k-auto-status" id="kAutoStatus">
        <div class="k-auto-msg" id="kAutoMsg"></div>
        <div class="k-auto-bar"><div class="k-auto-fill" id="kAutoFill"></div></div>
      </div>
      <button class="k-clear" id="kClearAll">&#215; Clear All Regions</button>
    </div>

    <div class="k-box">
      <div class="k-stitle">Export Karaoke Track</div>
      <div class="k-sub">Which vocalist should be silenced? (the one the karaoke singer will replace)</div>
      <div class="k-silence-row">
        <button class="k-sil-btn on" data-silence="Singer B" id="kSilB">Singer B</button>
        <button class="k-sil-btn" data-silence="Singer A" id="kSilA">Singer A</button>
        <button class="k-sil-btn" data-silence="Both" id="kSilBoth">Both Vocalists</button>
      </div>
      <div class="k-fmt-row">
        <select class="fmt-sel" id="kFmt"><option value="mp3">MP3</option><option value="wav">WAV</option><option value="flac">FLAC</option></select>
        <button class="k-exp-btn" id="kExpBtn">&#127908; Export Karaoke Track</button>
      </div>
    </div>

    <div class="k-result-box" id="kResultBox">
      <div class="k-result-hdr">&#10003; Karaoke track ready &mdash; preview below</div>
      <audio id="kAudio" controls preload="auto"></audio>
      <a class="k-dl" id="kDlBtn" download="karaoke.mp3">&#11015; Download</a>
    </div>

    <button class="rst-btn" id="kRstBtn">&#8617; Process Another Track</button>
  </div>

  <div class="steps" id="stepsRow">
    <div class="step"><div class="step-n">Step 01</div><span class="step-i">&#128193;</span><h4>Upload Track</h4><p>MP3, WAV, FLAC and more</p></div>
    <div class="step"><div class="step-n">Step 02</div><span class="step-i">&#127911;</span><h4>Clean &amp; Master</h4><p>Remove artifacts, remaster stems</p></div>
    <div class="step"><div class="step-n">Step 03</div><span class="step-i">&#10024;</span><h4>Mix &amp; AI Remix</h4><p>Mix, export, or generate with AI</p></div>
  </div>
</main>
<footer>StemSplit runs locally on your machine. Your audio never leaves your computer.</footer>

<script>
(function() {
  var model='htdemucs', jobId=null, poll=null, state={}, trackDur=30;
  var trackPassNum=1, estPasses=2;
  var wsInstance=null, regionData={}, kCurrentLabel='Singer A', kSilenceLabel='Singer B';
  var aiMode='generate', aiJobId=null, aiPoll=null, aiOpen=false;
  var cleanIntensity='medium', cleanPoll=null;

  var ICONS={vocals:'&#127908;',drums:'&#129345;',bass:'&#127926;',other:'&#127925;',guitar:'&#127928;',piano:'&#127929;',no_vocals:'&#127932;'};
  var INT_DESCS={
    light: 'Gentle noise reduction (30%), subtle EQ, light 2:1 compression. Best for cleaner sources.',
    medium:'Moderate noise reduction (55%), stem-specific EQ, 3.5:1 compression, loudness normalization. Good for most tracks.',
    heavy: 'Aggressive noise reduction (80%), corrective EQ, heavy 6:1 compression. Use for noisy or artifact-heavy separations.'
  };
  var MODE_META={
    generate: {desc:'Generate entirely new music from your text description',label:'&#10024; Generate Fresh'},
    transfer: {desc:'Transform your uploaded track into a new style while preserving melody and structure',label:'&#128260; Style Transfer'},
    reference:{desc:'Generate new music that captures the acoustic character of your track',label:'&#127911; Reference Style'}
  };

  // ── MODEL ──
  document.getElementById('m4').addEventListener('click',function(){this.classList.add('on');document.getElementById('m6').classList.remove('on');document.getElementById('mKar').classList.remove('on');model='htdemucs';});
  document.getElementById('m6').addEventListener('click',function(){this.classList.add('on');document.getElementById('m4').classList.remove('on');document.getElementById('mKar').classList.remove('on');model='htdemucs_6s';});
  document.getElementById('mKar').addEventListener('click',function(){this.classList.add('on');document.getElementById('m4').classList.remove('on');document.getElementById('m6').classList.remove('on');model='karaoke';});

  // ── VIEWS ──
  function show(v){
    document.getElementById('view-upload').style.display    =v==='upload'    ?'block':'none';
    document.getElementById('view-processing').style.display=v==='processing'?'flex' :'none';
    document.getElementById('view-results').style.display   =v==='results'   ?'flex' :'none';
    var _vk=document.getElementById('view-karaoke');if(_vk)_vk.style.display=v==='karaoke'?'flex':'none';
    document.getElementById('stepsRow').style.display       =v==='upload'    ?'flex' :'none';
  }

  // ── UPLOAD ──
  var fileInput=document.getElementById('fileInput'), dropZone=document.getElementById('dropZone');
  document.getElementById('pickBtn').addEventListener('click',function(e){e.stopPropagation();fileInput.click();});
  fileInput.addEventListener('change',function(){if(this.files&&this.files[0])go(this.files[0]);});
  dropZone.addEventListener('dragover',function(e){e.preventDefault();this.classList.add('over');});
  dropZone.addEventListener('dragleave',function(){this.classList.remove('over');});
  dropZone.addEventListener('drop',function(e){e.preventDefault();this.classList.remove('over');if(e.dataTransfer.files[0])go(e.dataTransfer.files[0]);});

  function go(file){
    clearErr();show('processing');
    document.getElementById('proc-name').textContent=file.name;
    document.getElementById('proc-msg').textContent='Uploading...';
    trackPassNum=1; estPasses=2;
    var pw=document.getElementById('progWrap');
    var pf=document.getElementById('progFill');
    var pp=document.getElementById('progPct');
    if(pw)pw.style.display='none';
    if(pf)pf.style.width='0%';
    if(pp)pp.textContent='0%';
    var fd=new FormData();fd.append('file',file);fd.append('model',model);
    fetch('/api/separate',{method:'POST',body:fd})
      .then(function(r){return r.json();})
      .then(function(d){if(d.error){err(d.error);return;}jobId=d.job_id;startPoll(file.name);})
      .catch(function(e){err('Server error: '+e.message);});
  }

  function startPoll(fname){
    poll=setInterval(function(){
      fetch('/api/status/'+jobId)
        .then(function(r){return r.json();})
        .then(function(d){
          document.getElementById('proc-msg').textContent=d.message||'Working...';
          var pct=d.progress||0;
          var passNum=d.pass_num||1;
          // Update pass estimate dynamically
          if(passNum>trackPassNum){trackPassNum=passNum;estPasses=Math.max(estPasses,passNum+1);}
          // Slot-based progress: each pass occupies 1/estPasses of the bar
          var slotStart=Math.round((passNum-1)/estPasses*100);
          var slotEnd  =Math.round(passNum/estPasses*100)-1;
          var displayPct=d.status==='done'?100:Math.round(slotStart+(pct/100)*(slotEnd-slotStart));
          if(displayPct>0||pct>0){
            var _pw=document.getElementById('progWrap');
            var _pf=document.getElementById('progFill');
            var _pp=document.getElementById('progPct');
            if(_pw)_pw.style.display='flex';
            if(_pf)_pf.style.width=displayPct+'%';
            if(_pp)_pp.textContent='Step '+passNum+' • '+displayPct+'%';
          }
          if(d.status==='done'){clearInterval(poll);trackDur=d.duration||30;if(model==='karaoke'){buildKaraokeView(d.stems,fname);}else{buildMixer(d.stems,fname);}var sb=document.getElementById('pjSaveBtn');if(sb)sb.style.display='inline-block';pjLoad();}
          if(d.status==='error'){clearInterval(poll);err(d.message);}
        }).catch(function(){});
    },1000);
  }

  // ── MIXER ──
  function buildMixer(stems,fname){
    show('results');
    document.getElementById('resFile').textContent=fname;
    document.getElementById('abBox').style.display='none';
    document.getElementById('origAudio').src='/api/original/'+jobId;
    document.getElementById('aiDur').value=Math.round(trackDur);
    document.getElementById('aiDur2').value=Math.round(trackDur);
    resetCleanUI();
    checkAiStatus();
    state={};
    stems.forEach(function(s){state[s.name]={vol:100,muted:false,soloed:false};});
    var grid=document.getElementById('stemsGrid');grid.innerHTML='';
    stems.forEach(function(s){
      var card=document.createElement('div');card.className='stem-card';card.id='c-'+s.name;
      var top=document.createElement('div');top.className='stem-top';
      var mb=document.createElement('button');mb.className='cbtn mute-btn';mb.id='m-'+s.name;mb.title='Mute';mb.textContent='M';
      mb.addEventListener('click',(function(n){return function(){toggleMute(n);};})(s.name));
      var sb=document.createElement('button');sb.className='cbtn solo-btn';sb.id='s-'+s.name;sb.title='Solo';sb.textContent='S';
      sb.addEventListener('click',(function(n){return function(){toggleSolo(n);};})(s.name));
      top.appendChild(mb);top.appendChild(sb);
      var mid=document.createElement('div');mid.className='stem-mid';
      var ico=document.createElement('span');ico.className='stem-ico';ico.innerHTML=ICONS[s.name]||'&#127925;';
      var nm=document.createElement('div');nm.className='stem-nm';nm.textContent=s.name;
      var tag=document.createElement('span');tag.className='stem-clean-tag';tag.id='ct-'+s.name;
      mid.appendChild(ico);mid.appendChild(nm);mid.appendChild(tag);
      var vw=document.createElement('div');vw.className='vol-wrap';
      var sl=document.createElement('input');sl.type='range';sl.className='vol-sl';sl.id='v-'+s.name;sl.min=0;sl.max=100;sl.value=100;
      sl.addEventListener('input',(function(n){return function(){setVol(n,this.value);};})(s.name));
      var lb=document.createElement('div');lb.className='vol-lbl';lb.id='vl-'+s.name;lb.textContent='100%';
      vw.appendChild(sl);vw.appendChild(lb);
      // AI Replace section
      var aiBadge=document.createElement('div');aiBadge.className='ai-badge-rep';aiBadge.id='airb-'+s.name;aiBadge.textContent='AI ACTIVE';
      mid.appendChild(aiBadge);
      var aiSec=document.createElement('div');aiSec.className='ai-rep-sec';
      var aiTog=document.createElement('button');aiTog.className='ai-rep-toggle';aiTog.id='airt-'+s.name;
      aiTog.innerHTML='&#10024; Replace with AI';
      var aiPan=document.createElement('div');aiPan.className='ai-rep-panel';aiPan.id='airp-'+s.name;
      var aiInp=document.createElement('textarea');aiInp.className='ai-rep-inp';aiInp.id='airi-'+s.name;
      aiInp.rows=2;aiInp.placeholder=AI_HINTS[s.name]||'Describe the replacement style...';
      var aiGen=document.createElement('button');aiGen.className='ai-rep-gen';aiGen.id='airg-'+s.name;
      aiGen.innerHTML='&#10024; Generate Replacement';
      aiGen.addEventListener('click',(function(n){return function(){generateStemRep(n);};})(s.name));
      var aiMsg=document.createElement('div');aiMsg.className='ai-rep-msg';aiMsg.id='airm-'+s.name;
      var aiRev=document.createElement('button');aiRev.className='ai-rep-revert';aiRev.id='airv-'+s.name;
      aiRev.style.display='none';aiRev.innerHTML='&#8617; Revert to Original';
      aiRev.addEventListener('click',(function(n){return function(){revertStemRep(n);};})(s.name));
      aiPan.appendChild(aiInp);aiPan.appendChild(aiGen);aiPan.appendChild(aiMsg);aiPan.appendChild(aiRev);
      aiSec.appendChild(aiTog);aiSec.appendChild(aiPan);
      aiTog.addEventListener('click',(function(pan,tog){return function(){
        var open=pan.style.display!=='none';
        pan.style.display=open?'none':'flex';
        tog.classList.toggle('on',!open);
      };})(aiPan,aiTog));
      card.appendChild(top);card.appendChild(mid);card.appendChild(vw);card.appendChild(aiSec);
      // Vocal split button — only on the vocals stem
      if(s.name==='vocals'){
        var svSec=document.createElement('div');svSec.className='sv-sec';
        var svBtn=document.createElement('button');svBtn.className='sv-btn';svBtn.id='svbtn';
        svBtn.innerHTML='&#10024; Split Lead / Backing Vocals';
        svBtn.addEventListener('click', splitVocals);
        var svMsg=document.createElement('div');svMsg.className='sv-msg';svMsg.id='svmsg';
        svSec.appendChild(svBtn);svSec.appendChild(svMsg);
        card.appendChild(svSec);
      }
      grid.appendChild(card);fillSlider(sl,100);
    });
  }


  // ── STEM AI REPLACEMENT ─────────────────────────────────────────────
  var AI_HINTS = {
    lead_vocals:    'e.g. powerful lead vocal, clear and upfront',
    backing_vocals: 'e.g. soft harmony vocals, lush and reverb-heavy',
    vocals:    'e.g. soulful female vocal, smooth R&B, melodic',
    drums:     'e.g. trap drums, 808 kick, crisp hi-hats, 90 BPM',
    bass:      'e.g. deep synth bass, warm sub, punchy and funky',
    guitar:    'e.g. clean electric guitar, fingerpicked arpeggios',
    piano:     'e.g. soft piano, melancholic, sparse, cinematic',
    other:     'e.g. lush string pad, atmospheric, wide and spacious',
    no_vocals: 'e.g. full orchestral arrangement, same tempo',
  };

  function getAiReady() {
    var b=document.getElementById('aiBadge');
    return b && b.classList.contains('ready');
  }


  // ── VOCAL SPLIT (lead vs backing) ───────────────────────────────────
  function splitVocals() {
    var btn=document.getElementById('svbtn');
    var msg=document.getElementById('svmsg');
    if(!btn)return;
    btn.disabled=true; btn.textContent='Splitting...';
    if(msg)msg.textContent='Starting vocal split...';

    fetch('/api/split_vocals/'+jobId,{method:'POST'})
      .then(function(r){return r.json();})
      .then(function(d){
        if(d.error){
          btn.disabled=false; btn.innerHTML='&#10024; Split Lead / Backing Vocals';
          if(msg)msg.textContent='Error: '+d.error; return;
        }
        pollVocalSplit(d.split_job_id);
      })
      .catch(function(e){
        btn.disabled=false; btn.innerHTML='&#10024; Split Lead / Backing Vocals';
        if(msg)msg.textContent='Error: '+e.message;
      });
  }

  function pollVocalSplit(sjid) {
    var msg=document.getElementById('svmsg');
    var splitPoll=setInterval(function(){
      fetch('/api/vocal_split_status/'+sjid)
        .then(function(r){return r.json();})
        .then(function(d){
          if(msg)msg.textContent=d.message||'Processing...';
          if(d.status==='done'){
            clearInterval(splitPoll);
            // Rebuild mixer with updated stems (vocals → lead_vocals + backing_vocals)
            fetch('/api/status/'+jobId)
              .then(function(r){return r.json();})
              .then(function(sd){
                if(sd.stems){
                  var fname=document.getElementById('resFile').textContent||'track';
                  buildMixer(sd.stems, fname);
                }
              });
          }else if(d.status==='error'){
            clearInterval(splitPoll);
            var btn2=document.getElementById('svbtn');
            if(btn2){btn2.disabled=false;btn2.innerHTML='&#10024; Split Lead / Backing Vocals';}
            if(msg)msg.textContent='Failed: '+d.message;
          }
        }).catch(function(){});
    },2500);
  }

  function generateStemRep(n) {
    if (!getAiReady()) {
      err('AceStep worker is not running. Launch with start_all.bat and wait for the AI Remix panel to show "Ready".');
      return;
    }
    var prompt=document.getElementById('airi-'+n).value.trim();
    if(!prompt){document.getElementById('airi-'+n).focus();return;}
    var gen=document.getElementById('airg-'+n), msg=document.getElementById('airm-'+n);
    gen.disabled=true; gen.textContent='Generating...'; msg.textContent='Starting...';
    if(!state[n].muted) toggleMute(n);
    fetch('/api/replace_stem/'+jobId+'/'+n,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt:prompt})
    })
    .then(function(r){return r.json();})
    .then(function(d){
      if(d.error){gen.disabled=false;gen.innerHTML='&#10024; Generate Replacement';msg.textContent='Error: '+d.error;return;}
      state[n].aiJobId=d.replace_job_id;
      pollStemRep(n,d.replace_job_id);
    })
    .catch(function(e){gen.disabled=false;gen.innerHTML='&#10024; Generate Replacement';msg.textContent='Error: '+e.message;});
  }

  function pollStemRep(n,rjid) {
    if(state[n].aiPoll)clearInterval(state[n].aiPoll);
    state[n].aiPoll=setInterval(function(){
      fetch('/api/replace_status/'+rjid)
        .then(function(r){return r.json();})
        .then(function(d){
          var msg=document.getElementById('airm-'+n);
          var gen=document.getElementById('airg-'+n);
          var rev=document.getElementById('airv-'+n);
          var bdg=document.getElementById('airb-'+n);
          var crd=document.getElementById('c-'+n);
          var tog=document.getElementById('airt-'+n);
          if(msg)msg.textContent=d.message||'Working...';
          if(d.status==='done'){
            clearInterval(state[n].aiPoll); state[n].aiActive=true;
            if(gen){gen.disabled=false;gen.innerHTML='&#10024; Regenerate';}
            if(rev)rev.style.display='block';
            if(bdg)bdg.style.display='block';
            if(crd)crd.classList.add('ai-replaced');
            if(tog)tog.classList.add('on');
            if(msg)msg.textContent='AI active \u2014 unmute to include in export';
            if(state[n].muted)toggleMute(n);
          } else if(d.status==='error'){
            clearInterval(state[n].aiPoll);
            if(gen){gen.disabled=false;gen.innerHTML='&#10024; Try Again';}
            if(msg)msg.textContent='Failed: '+d.message;
          }
        }).catch(function(){});
    },3000);
  }

  function revertStemRep(n) {
    state[n].aiActive=false;
    if(state[n].aiPoll)clearInterval(state[n].aiPoll);
    var crd=document.getElementById('c-'+n), tog=document.getElementById('airt-'+n);
    var bdg=document.getElementById('airb-'+n), rev=document.getElementById('airv-'+n);
    var gen=document.getElementById('airg-'+n), msg=document.getElementById('airm-'+n);
    if(crd)crd.classList.remove('ai-replaced');
    if(tog)tog.classList.remove('on');
    if(bdg)bdg.style.display='none';
    if(rev)rev.style.display='none';
    if(gen){gen.disabled=false;gen.innerHTML='&#10024; Generate Replacement';}
    if(msg)msg.textContent='';
  }

  function toggleMute(n){state[n].muted=!state[n].muted;if(state[n].muted)state[n].soloed=false;refresh();clearPre();}
  function toggleSolo(n){
    var was=state[n].soloed;Object.keys(state).forEach(function(k){state[k].soloed=false;});
    if(!was){state[n].soloed=true;state[n].muted=false;}
    refresh();clearPre();
  }
  function setVol(n,v){state[n].vol=parseInt(v);document.getElementById('vl-'+n).textContent=v+'%';fillSlider(document.getElementById('v-'+n),v);clearPre();}
  function fillSlider(sl,v){sl.style.background='linear-gradient(to right,#7C3AED '+v+'%,rgba(255,255,255,.1) '+v+'%)';}
  function refresh(){
    var solo=Object.keys(state).some(function(k){return state[k].soloed;});
    Object.keys(state).forEach(function(n){
      var s=state[n],aud=solo?s.soloed:!s.muted;
      var card=document.getElementById('c-'+n),mb=document.getElementById('m-'+n),sb=document.getElementById('s-'+n);
      if(card){card.classList.toggle('muted',!aud);card.classList.toggle('soloed',s.soloed);}
      if(mb)mb.classList.toggle('on',s.muted&&!solo);
      if(sb)sb.classList.toggle('on',s.soloed);
    });
  }

  // ── PRESETS ──
  document.querySelectorAll('.pre-btn').forEach(function(btn){
    btn.addEventListener('click',function(){applyPreset(this.dataset.preset,this);});
  });
  function applyPreset(p,btn){
    clearPre();if(btn)btn.classList.add('on');
    Object.keys(state).forEach(function(n){
      state[n].soloed=false;state[n].vol=100;
      if(p==='karaoke')      state[n].muted=(n==='vocals'||n==='lead_vocals'||n==='backing_vocals');
      else if(p==='acapella')state[n].muted=(n!=='vocals'&&n!=='lead_vocals'&&n!=='backing_vocals');
      else if(p==='instrumental')state[n].muted=(n==='vocals');
      else if(p==='drumsandbass')state[n].muted=(n!=='drums'&&n!=='bass');
      else state[n].muted=false;
      var sl=document.getElementById('v-'+n),lb=document.getElementById('vl-'+n);
      if(sl){sl.value=100;fillSlider(sl,100);}if(lb)lb.textContent='100%';
    });
    refresh();
  }
  function clearPre(){document.querySelectorAll('.pre-btn').forEach(function(b){b.classList.remove('on');});}

  // ── EXPORT ──
  document.getElementById('expBtn').addEventListener('click',function(){
    var btn=this,fmt=document.getElementById('fmtSel').value;
    btn.disabled=true;btn.textContent='Exporting...';
    var solo=Object.keys(state).some(function(k){return state[k].soloed;});
    var payload={};
    Object.keys(state).forEach(function(n){payload[n]={volume:state[n].vol,muted:solo?!state[n].soloed:state[n].muted,use_ai_replacement:state[n].aiActive===true};});
    fetch('/api/mix/'+jobId,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({stems:payload,format:fmt})})
      .then(function(r){if(!r.ok)return r.json().then(function(d){throw new Error(d.error||'Export failed');});return r.blob();})
      .then(function(blob){
        var url=URL.createObjectURL(blob);
        var a=document.createElement('a');a.href=url;a.download='mix.'+fmt;a.click();
        document.getElementById('mixAudio').src=url;
        document.getElementById('abBox').style.display='block';
      })
      .catch(function(e){err(e.message||'Export failed.');})
      .finally(function(){btn.disabled=false;btn.innerHTML='&#11015; Export Mix';});
  });

  // ── CLEAN & MASTER ──
  document.querySelectorAll('.int-btn').forEach(function(btn){
    btn.addEventListener('click',function(){
      cleanIntensity=this.dataset.val;
      document.querySelectorAll('.int-btn').forEach(function(b){b.classList.remove('on');});
      this.classList.add('on');
      document.getElementById('intDesc').textContent=INT_DESCS[cleanIntensity];
    });
  });

  document.getElementById('cleanBtn').addEventListener('click',function(){
    document.getElementById('cleanIdle').style.display='none';
    document.getElementById('cleanProg').style.display='flex';
    document.getElementById('cleanMsg').textContent='Starting cleanup...';

    fetch('/api/clean/'+jobId,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({intensity:cleanIntensity})})
      .then(function(r){return r.json();})
      .then(function(d){
        if(d.error){showCleanErr(d.error);return;}
        pollClean();
      })
      .catch(function(e){showCleanErr(e.message);});
  });

  function pollClean(){
    cleanPoll=setInterval(function(){
      fetch('/api/clean_status/'+jobId)
        .then(function(r){return r.json();})
        .then(function(d){
          document.getElementById('cleanMsg').textContent=d.message||'Processing...';
          if(d.status==='done'){
            clearInterval(cleanPoll);
            document.getElementById('cleanProg').style.display='none';
            document.getElementById('cleanDone').style.display='flex';
            // Mark stem cards as cleaned
            Object.keys(state).forEach(function(n){
              var card=document.getElementById('c-'+n);
              var tag=document.getElementById('ct-'+n);
              if(card)card.classList.add('cleaned');
              if(tag)tag.textContent='cleaned';
            });
          }
          if(d.status==='error'){clearInterval(cleanPoll);showCleanErr(d.message);}
        }).catch(function(){});
    },2500);
  }

  function showCleanErr(msg){
    document.getElementById('cleanProg').style.display='none';
    document.getElementById('cleanIdle').style.display='flex';
    err('Clean & Master: '+msg);
  }

  function resetCleanUI(){
    if(cleanPoll)clearInterval(cleanPoll);
    document.getElementById('cleanIdle').style.display='flex';
    document.getElementById('cleanProg').style.display='none';
    document.getElementById('cleanDone').style.display='none';
    document.getElementById('cleanBtn').disabled=false;
    cleanIntensity='medium';
    document.querySelectorAll('.int-btn').forEach(function(b){b.classList.toggle('on',b.dataset.val==='medium');});
    document.getElementById('intDesc').textContent=INT_DESCS.medium;
  }

  // ── AI REMIX ──
  function toggleAiPanel(){
    aiOpen=!aiOpen;
    document.getElementById('aiBody').style.display=aiOpen?'flex':'none';
    document.getElementById('aiChevron').classList.toggle('open',aiOpen);
    if(aiOpen)checkAiStatus();
  }
  document.getElementById('aiHdr').addEventListener('click',function(e){if(e.target.tagName!=='BUTTON')toggleAiPanel();});
  document.getElementById('aiChevron').addEventListener('click',function(e){e.stopPropagation();toggleAiPanel();});

  function setAiMode(m){
    aiMode=m;
    ['generate','transfer','reference'].forEach(function(id){
      document.getElementById('aiMode'+id.charAt(0).toUpperCase()+id.slice(1)).classList.toggle('on',id===m);
    });
    document.getElementById('aiModeDesc').textContent=MODE_META[m].desc;
    document.getElementById('aiFreshCtrl').style.display   =m==='generate' ?'flex':'none';
    document.getElementById('aiTransferCtrl').style.display=m==='transfer' ?'flex':'none';
    document.getElementById('aiReferenceCtrl').style.display=m==='reference'?'flex':'none';
    document.getElementById('aiGenBtn').innerHTML=MODE_META[m].label;
  }

  document.getElementById('aiModeGenerate').addEventListener('click',function(){setAiMode('generate');});
  document.getElementById('aiModeTransfer').addEventListener('click',function(){setAiMode('transfer');});
  document.getElementById('aiModeReference').addEventListener('click',function(){setAiMode('reference');});

  document.getElementById('aiStrength').addEventListener('input',function(){
    var val=(parseInt(this.value)/10).toFixed(1);
    document.getElementById('aiStrengthVal').textContent=val;
    fillSlider(this,parseInt(this.value)*10);
  });
  fillSlider(document.getElementById('aiStrength'),40);

  function checkAiStatus(){
    fetch('/api/ai/status')
      .then(function(r){return r.json();})
      .then(function(d){
        if(d.ready)        {setBadge('ready','Ready');showAiState('ready');}
        else if(d.loading) {setBadge('loading','Loading models...');showAiState('loading');setTimeout(checkAiStatus,5000);}
        else               {setBadge('offline','Offline');showAiState('offline');}
      }).catch(function(){setBadge('offline','Offline');showAiState('offline');});
  }
  function setBadge(cls,txt){var b=document.getElementById('aiBadge');b.className='ai-badge '+cls;b.textContent=txt;}
  function showAiState(s){
    document.getElementById('aiOffline').style.display   =s==='offline'   ?'block':'none';
    document.getElementById('aiLoading').style.display   =s==='loading'   ?'flex' :'none';
    document.getElementById('aiReady').style.display     =s==='ready'     ?'flex' :'none';
    document.getElementById('aiGenerating').style.display=s==='generating'?'flex' :'none';
    document.getElementById('aiResult').style.display    =s==='result'    ?'flex' :'none';
  }

  document.getElementById('aiGenBtn').addEventListener('click',function(){
    var prompt=document.getElementById('aiPrompt').value.trim();
    if(!prompt){document.getElementById('aiPrompt').focus();return;}
    showAiState('generating');
    document.getElementById('aiProgMsg').textContent='Sending to AI...';
    var endpoint,body;
    if(aiMode==='generate'){
      endpoint='/api/ai/generate';
      body={prompt:prompt,duration:parseFloat(document.getElementById('aiDur').value)||30,bpm:document.getElementById('aiBpm').value.trim()||null};
    }else if(aiMode==='transfer'){
      endpoint='/api/ai/cover';
      body={stem_job_id:jobId,prompt:prompt,strength:parseInt(document.getElementById('aiStrength').value)/10,duration:parseFloat(document.getElementById('aiCoverDur').value)||120};
    }else{
      endpoint='/api/ai/reference';
      body={stem_job_id:jobId,prompt:prompt,duration:parseFloat(document.getElementById('aiDur2').value)||trackDur,bpm:document.getElementById('aiBpm2').value.trim()||null};
    }
    fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
      .then(function(r){return r.json();})
      .then(function(d){if(d.error){showAiState('ready');err('AI: '+d.error);return;}aiJobId=d.job_id;pollAiJob();})
      .catch(function(e){showAiState('ready');err('AI request failed: '+e.message);});
  });

  function pollAiJob(){
    aiPoll=setInterval(function(){
      fetch('/api/ai/job/'+aiJobId)
        .then(function(r){return r.json();})
        .then(function(d){
          document.getElementById('aiProgMsg').textContent=d.message||'Generating...';
          if(d.status==='done'){
            clearInterval(aiPoll);
            var url='/api/ai/download/'+aiJobId;
            document.getElementById('aiAudio').src=url;
            document.getElementById('aiDlBtn').href=url;
            var labels={generate:'&#10024; AI Remix ready',transfer:'&#128260; Style Transfer ready',reference:'&#127911; Reference Style ready'};
            document.getElementById('aiResHdr').innerHTML=labels[aiMode]||'&#10003; AI Remix ready';
            showAiState('result');
          }else if(d.status==='error'){clearInterval(aiPoll);showAiState('ready');err('AI failed: '+d.message);}
        }).catch(function(){});
    },3000);
  }

  document.getElementById('aiAgainBtn').addEventListener('click',function(){if(aiPoll)clearInterval(aiPoll);showAiState('ready');});

  // ── RESET ──
  document.getElementById('rstBtn').addEventListener('click',resetAll);
  function resetAll(){
    if(poll)clearInterval(poll);if(aiPoll)clearInterval(aiPoll);if(cleanPoll)clearInterval(cleanPoll);
    if(wsInstance){try{wsInstance.destroy();}catch(e){}wsInstance=null;}
    clearErr();fileInput.value='';jobId=null;state={};regionData={};
    document.getElementById('abBox').style.display='none';
    aiOpen=false;document.getElementById('aiBody').style.display='none';
    document.getElementById('aiChevron').classList.remove('open');
    show('upload');
  }

  // ── KARAOKE EDITOR ──────────────────────────────────────────────────
  var K_COLORS = {
    'Singer A':'rgba(124,58,237,0.30)',
    'Singer B':'rgba(16,185,129,0.30)',
    'Both':    'rgba(245,158,11,0.30)'
  };
  var K_DOT = {'Singer A':'#7C3AED','Singer B':'#10B981','Both':'#F59E0B'};
  var kDrawMode = true;

  function buildKaraokeView(stems, fname) {
    show('karaoke');
    document.getElementById('kFile').textContent = fname;
    document.getElementById('kResultBox').style.display = 'none';
    regionData={}; kCurrentLabel='Singer A'; kSilenceLabel='Singer B'; kDrawMode=true;
    var dm=document.getElementById('wfDrawMode'),sm=document.getElementById('wfSeekMode');
    if(dm)dm.classList.add('on'); if(sm)sm.classList.remove('on');
    var zsl=document.getElementById('wfZoom'); if(zsl)zsl.value=0;
    document.querySelectorAll('.k-label-btn').forEach(function(b){b.classList.toggle('on',b.dataset.klabel==='Singer A');});
    document.querySelectorAll('.k-sil-btn').forEach(function(b){b.classList.toggle('on',b.dataset.silence==='Singer B');});
    document.getElementById('kRegionsList').innerHTML='<div class="k-regions-empty">No regions yet. In Draw mode, drag on the waveform to mark a segment.</div>';
    if(wsInstance){try{wsInstance.destroy();}catch(e){}wsInstance=null;}
    if(typeof WaveSurfer==='undefined'){
      document.getElementById('kFile').textContent=fname+' - WaveSurfer not loaded'; return;
    }
    wsInstance=WaveSurfer.create({
      container:'#waveform', waveColor:'#7C3AED', progressColor:'#A78BFA',
      cursorColor:'rgba(255,255,255,0.8)', cursorWidth:2, height:90,
      responsive:true, normalize:true, interact:true, scrollParent:true, hideScrollbar:false,
      plugins:[
        WaveSurfer.regions.create({regionsMinLength:0.1}),
        WaveSurfer.timeline.create({
          container:'#wf-timeline',
          primaryColor:'rgba(255,255,255,0.4)', secondaryColor:'rgba(255,255,255,0.2)',
          primaryFontColor:'rgba(255,255,255,0.5)', secondaryFontColor:'rgba(255,255,255,0.3)',
          fontFamily:'Inter,system-ui,sans-serif', fontSize:10
        })
      ]
    });
    wsInstance.load('/api/stem_audio/'+jobId+'/vocals');
    wsInstance.on('ready', function(){
      wsInstance.enableDragSelection({color:K_COLORS[kCurrentLabel]});
      updateWfTime();
    });
    wsInstance.on('audioprocess', updateWfTime);
    wsInstance.on('seek', updateWfTime);
    wsInstance.on('pause',  function(){document.getElementById('wfPlay').innerHTML='&#9654;';});
    wsInstance.on('finish', function(){document.getElementById('wfPlay').innerHTML='&#9654;';});
    wsInstance.on('play',   function(){document.getElementById('wfPlay').innerHTML='&#9646;&#9646;';});
    wsInstance.on('region-created', function(region){
      if(!kDrawMode){region.remove();return;}
      regionData[region.id]={label:kCurrentLabel};
      region.update({color:K_COLORS[kCurrentLabel]});
      addWfLabel(region, kCurrentLabel);
      updateKRegionsList();
    });
    wsInstance.on('region-updated', function(){updateKRegionsList();});
    wsInstance.on('region-removed', function(region){delete regionData[region.id]; updateKRegionsList();});
  }

  function addWfLabel(region, label) {
    var el=region.element;
    el.style.overflow='hidden';
    el.style.borderTop='3px solid '+K_DOT[label];
    var span=el.querySelector('.k-wf-lbl');
    if(!span){
      span=document.createElement('span'); span.className='k-wf-lbl';
      span.style.cssText='position:absolute;top:4px;left:4px;font-size:10px;font-weight:700;color:#fff;padding:2px 6px;border-radius:3px;pointer-events:none;white-space:nowrap;overflow:hidden;max-width:calc(100% - 8px);text-overflow:ellipsis;z-index:2';
      el.appendChild(span);
    }
    span.textContent=label; span.style.background=K_DOT[label];
  }

  function changeRegionLabel(rid, newLabel) {
    if(!wsInstance||!regionData[rid]) return;
    regionData[rid].label=newLabel;
    var region=wsInstance.regions.list[rid];
    if(region){region.update({color:K_COLORS[newLabel]}); addWfLabel(region,newLabel);}
    var sw=document.getElementById('ksw-'+rid); if(sw)sw.style.background=K_DOT[newLabel];
  }

  function updateWfTime() {
    if(!wsInstance) return;
    var cur=wsInstance.getCurrentTime(), dur=wsInstance.getDuration();
    document.getElementById('wfTime').textContent=fmtT(cur)+' / '+fmtT(dur);
  }

  function fmtT(s) {
    var m=Math.floor(s/60), sec=Math.floor(s%60), ds=Math.floor((s%1)*10);
    return m+':'+(sec<10?'0':'')+sec+'.'+ds;
  }

  function updateKRegionsList() {
    if(!wsInstance) return;
    var regions=wsInstance.regions.list, ids=Object.keys(regions);
    var list=document.getElementById('kRegionsList');
    if(ids.length===0){
      list.innerHTML='<div class="k-regions-empty">No regions yet. In Draw mode, drag on the waveform to mark a segment.</div>';
      return;
    }
    ids.sort(function(a,b){return regions[a].start-regions[b].start;});
    list.innerHTML='';
    ids.forEach(function(id){
      var r=regions[id], d=regionData[id]||{label:'Singer A'};
      var item=document.createElement('div'); item.className='k-region-item';
      var sw=document.createElement('div'); sw.className='k-region-swatch'; sw.id='ksw-'+id;
      sw.style.background=K_DOT[d.label]||'#888';
      var ti=document.createElement('div'); ti.className='k-region-times';
      ti.textContent=fmtT(r.start)+' \u2192 '+fmtT(r.end);
      var sel=document.createElement('select'); sel.className='k-region-sel';
      ['Singer A','Singer B','Both'].forEach(function(opt){
        var o=document.createElement('option'); o.value=opt; o.textContent=opt;
        if(opt===d.label)o.selected=true; sel.appendChild(o);
      });
      sel.addEventListener('change',(function(rid2){return function(){changeRegionLabel(rid2,this.value);};})(id));
      var del=document.createElement('button'); del.className='k-region-del'; del.innerHTML='\u00d7'; del.title='Remove';
      del.addEventListener('click',(function(rid2){return function(){
        var reg=wsInstance.regions.list[rid2]; if(reg)reg.remove();
      };})(id));
      item.appendChild(sw); item.appendChild(ti); item.appendChild(sel); item.appendChild(del);
      list.appendChild(item);
    });
  }

  document.querySelectorAll('.k-label-btn').forEach(function(btn){
    btn.addEventListener('click', function(){
      kCurrentLabel=this.dataset.klabel;
      document.querySelectorAll('.k-label-btn').forEach(function(b){b.classList.remove('on');});
      this.classList.add('on');
      if(wsInstance&&kDrawMode) wsInstance.enableDragSelection({color:K_COLORS[kCurrentLabel]});
    });
  });

  document.querySelectorAll('.k-sil-btn').forEach(function(btn){
    btn.addEventListener('click', function(){
      kSilenceLabel=this.dataset.silence;
      document.querySelectorAll('.k-sil-btn').forEach(function(b){b.classList.remove('on');});
      this.classList.add('on');
    });
  });

  document.getElementById('wfDrawMode').addEventListener('click', function(){
    kDrawMode=true; this.classList.add('on');
    document.getElementById('wfSeekMode').classList.remove('on');
    if(wsInstance) wsInstance.enableDragSelection({color:K_COLORS[kCurrentLabel]});
  });
  document.getElementById('wfSeekMode').addEventListener('click', function(){
    kDrawMode=false; this.classList.add('on');
    document.getElementById('wfDrawMode').classList.remove('on');
    if(wsInstance) wsInstance.disableDragSelection();
  });

  document.getElementById('wfZoom').addEventListener('input', function(){
    if(wsInstance) wsInstance.zoom(parseInt(this.value));
  });
  document.getElementById('wfZoomReset').addEventListener('click', function(){
    if(wsInstance){wsInstance.zoom(0); document.getElementById('wfZoom').value=0;}
  });

  document.getElementById('wfPlay').addEventListener('click', function(){if(wsInstance)wsInstance.playPause();});
  document.getElementById('wfStop').addEventListener('click', function(){if(wsInstance)wsInstance.stop();});

  document.getElementById('kClearAll').addEventListener('click', function(){
    if(wsInstance){wsInstance.clearRegions(); regionData={}; updateKRegionsList();}
  });

  document.getElementById('kExpBtn').addEventListener('click', function(){
    var btn=this, fmt=document.getElementById('kFmt').value;
    var regions=wsInstance ? Object.values(wsInstance.regions.list).map(function(r){
      return {start:r.start,end:r.end,label:(regionData[r.id]||{}).label||'Singer A'};
    }) : [];
    btn.disabled=true; btn.textContent='Exporting...';
    document.getElementById('kResultBox').style.display='none';
    fetch('/api/karaoke_export/'+jobId,{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({regions:regions,silence_label:kSilenceLabel,format:fmt})
    })
    .then(function(r){if(!r.ok)return r.json().then(function(d){throw new Error(d.error||'Export failed');});return r.blob();})
    .then(function(blob){
      var url=URL.createObjectURL(blob);
      document.getElementById('kAudio').src=url; document.getElementById('kDlBtn').href=url;
      document.getElementById('kDlBtn').download='karaoke.'+fmt;
      document.getElementById('kResultBox').style.display='flex';
      document.getElementById('kResultBox').scrollIntoView({behavior:'smooth'});
    })
    .catch(function(e){err('Karaoke export: '+e.message);})
    .finally(function(){btn.disabled=false; btn.innerHTML='&#127908; Export Karaoke Track';});
  });

  document.getElementById('kRstBtn').addEventListener('click', resetAll);


  // ── YOUTUBE DOWNLOADER ──────────────────────────────────────────────
  var ytJobId = null;
  var ytPoll  = null;

  function ytCheckMusicUrl() {
    var url = document.getElementById('ytUrl').value.trim();
    var isMusic = url.indexOf('music.youtube.com') !== -1;
    document.getElementById('ytMusicHint').style.display = isMusic ? 'block' : 'none';
    if (isMusic) {
      var sel = document.getElementById('ytBrowser');
      if (!sel.value) sel.value = 'chrome';
    }
  }
  document.getElementById('ytUrl').addEventListener('input', ytCheckMusicUrl);
  document.getElementById('ytUrl').addEventListener('paste', function() {
    setTimeout(ytCheckMusicUrl, 0);
  });
  document.getElementById('ytUrl').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') document.getElementById('ytConvert').click();
  });
  document.getElementById('ytOpenBrowser').addEventListener('click', function() {
    fetch('/api/yt/open_login', {method:'POST'});
  });

  document.getElementById('ytConvert').addEventListener('click', function() {
    var url = document.getElementById('ytUrl').value.trim();
    if (!url) { document.getElementById('ytUrl').focus(); return; }
    var btn = this;
    btn.disabled = true; btn.textContent = 'Starting...';
    document.getElementById('ytStatus').style.display  = 'flex';
    document.getElementById('ytResult').style.display  = 'none';
    document.getElementById('ytProgFill').style.width  = '0%';
    document.getElementById('ytProgPct').textContent   = '0%';
    document.getElementById('ytMsg').textContent       = 'Connecting to YouTube...';
    if (ytPoll) clearInterval(ytPoll);

    fetch('/api/yt/download', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({url:url, browser:document.getElementById('ytBrowser').value})
    })
    .then(function(r){return r.json();})
    .then(function(d){
      if(d.error){
        btn.disabled=false; btn.textContent='Convert';
        document.getElementById('ytMsg').textContent='Error: '+d.error;
        return;
      }
      ytJobId=d.job_id;
      pollYtStatus(btn);
    })
    .catch(function(e){
      btn.disabled=false; btn.textContent='Convert';
      document.getElementById('ytMsg').textContent='Error: '+e.message;
    });
  });

  function pollYtStatus(btn) {
    ytPoll = setInterval(function() {
      fetch('/api/yt/status/'+ytJobId)
        .then(function(r){return r.json();})
        .then(function(d){
          var pct = d.progress || 0;
          document.getElementById('ytProgFill').style.width = pct+'%';
          document.getElementById('ytProgPct').textContent  = pct+'%';
          document.getElementById('ytMsg').textContent      = d.message || 'Working...';
          if (d.status === 'done') {
            clearInterval(ytPoll);
            btn.disabled=false; btn.textContent='Convert';
            document.getElementById('ytResultTitle').textContent = d.title || 'Download ready';
            var dlBtn = document.getElementById('ytDlBtn');
            dlBtn.href     = '/api/yt/file/'+ytJobId;
            dlBtn.download = d.filename || 'audio.mp3';
            document.getElementById('ytResult').style.display = 'flex';
          } else if (d.status === 'error') {
            clearInterval(ytPoll);
            btn.disabled=false; btn.textContent='Convert';
            var msg = d.message || 'Unknown error';
            var authMsg = 'Login required — select the browser where you are logged in to YouTube above, then try again.';
            var authWords = ['sign in','login','not a bot','403','private video','members only','premium','unavailable'];
            var lmsg = msg.toLowerCase();
            var isAuth = false;
            for (var i=0; i<authWords.length; i++) { if (lmsg.indexOf(authWords[i])!==-1) { isAuth=true; break; } }
            document.getElementById('ytMsg').textContent = 'Error: ' + (isAuth ? authMsg : msg);
            if (isAuth) document.getElementById('ytMusicHint').style.display = 'block';
          }
        }).catch(function(){});
    }, 1200);
  }

  document.getElementById('ytSepBtn').addEventListener('click', function() {
    var btn = this;
    btn.disabled = true; btn.textContent = 'Starting...';
    fetch('/api/yt/separate/'+ytJobId, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:model})
    })
    .then(function(r){return r.json();})
    .then(function(d){
      btn.disabled=false; btn.innerHTML='&#127925; Separate &amp; Remix';
      if(d.error){ err('YT Separate: '+d.error); return; }
      jobId=d.job_id;
      var title=document.getElementById('ytResultTitle').textContent||'YouTube track';
      clearErr(); show('processing');
      document.getElementById('proc-name').textContent=title;
      document.getElementById('proc-msg').textContent='Starting separation...';
      var pw=document.getElementById('progWrap'),pf=document.getElementById('progFill'),pp=document.getElementById('progPct');
      if(pw)pw.style.display='none'; if(pf)pf.style.width='0%'; if(pp)pp.textContent='0%';
      trackPassNum=1; estPasses=2;
      startPoll(title);
    })
    .catch(function(e){
      btn.disabled=false; btn.innerHTML='&#127925; Separate &amp; Remix';
      err('YT Separate: '+e.message);
    });
  });

  // ── PROJECT PANEL ──────────────────────────────────────────────────
  var _PJC=['#7C3AED','#10B981','#F59E0B','#EF4444','#3B82F6','#EC4899'];
  var _pjOpen=true;

  function _pjH(s){var h=0;for(var i=0;i<s.length;i++){h=((h<<5)-h)+s.charCodeAt(i);h|=0;}return h;}
  function _pjAge(iso){if(!iso)return '';var d=new Date(iso),n=new Date(),di=Math.floor((n-d)/1e3);if(di<60)return 'just now';if(di<3600)return Math.floor(di/60)+'m ago';if(di<86400)return Math.floor(di/3600)+'h ago';return Math.floor(di/86400)+'d ago';}
  function _pjDur(s){if(!s)return '';var m=Math.floor(s/60),sec=Math.floor(s%60);return m+':'+(sec<10?'0':'')+sec;}

  function pjToggle(){
    _pjOpen=!_pjOpen;
    var b=document.getElementById('pjBody'),c=document.getElementById('pjChev');
    if(b)b.className=_pjOpen?'pj-body':'pj-body closed';
    if(c)c.className=_pjOpen?'pj-chev open':'pj-chev';
  }

  function pjLoad(){
    var scr=document.getElementById('pjScroll');
    if(!scr)return;
    scr.innerHTML='<div style="padding:8px 4px;color:var(--t3);font-size:12px">Loading...</div>';
    fetch('/api/projects')
      .then(function(r){return r.json();})
      .then(function(ps){
        if(!ps||!ps.length){
          scr.innerHTML='<div class="pj-empty">No projects yet</div>';
          return;
        }
        scr.innerHTML='';
        ps.forEach(function(p){
          try{ scr.appendChild(_pjMakeCard(p)); }catch(ex){ console.error('Card error',ex); }
        });
      })
      .catch(function(e){
        scr.innerHTML='<div style="color:#F87171;padding:8px;font-size:12px">Could not load projects: '+e.message+'</div>';
      });
  }

  function _pjMakeCard(p){
    var card=document.createElement('div'); card.className='pj-card';
    var color=_PJC[Math.abs(_pjH(p.id||'x'))%_PJC.length];
    var mdl=p.model==='htdemucs_6s'?'6 stems':p.model==='karaoke'?'karaoke':'4 stems';
    var name=(p.name||'Untitled').replace(/</g,'&lt;');
    card.innerHTML='<div class="pj-bar" style="background:'+color+'"></div>'
      +'<div class="pj-body"><div class="pj-card-name">'+name+'</div>'
      +'<div class="pj-card-meta"><span>'+mdl+'</span>'
      +(p.duration?'<span>'+_pjDur(p.duration)+'</span>':'')
      +'<span>'+_pjAge(p.modified)+'</span></div></div>'
      +'<button class="pj-del">&#215;</button>';
    card.querySelector('.pj-del').addEventListener('click',function(e){
      e.stopPropagation();
      if(confirm('Remove from list? Audio files are NOT deleted.'))
        fetch('/api/projects/'+p.id,{method:'DELETE'}).then(pjLoad).catch(function(){});
    });
    card.addEventListener('click',function(){ pjOpenProject(p.id,p.name,p.model); });
    return card;
  }

  function pjOpenProject(pid,pname,pmodel){
    fetch('/api/projects/'+pid+'/load')
      .then(function(r){return r.json();})
      .then(function(d){
        if(d.error){alert('Could not load: '+d.error);return;}
        jobId=d.job_id; trackDur=d.duration||30; model=d.model||'htdemucs';
        ['m4','m6','mKar'].forEach(function(id){var el=document.getElementById(id);if(el)el.classList.remove('on');});
        var mb=document.getElementById(model==='karaoke'?'mKar':model==='htdemucs_6s'?'m6':'m4');
        if(mb)mb.classList.add('on');
        var sb=document.getElementById('pjSaveBtn');if(sb)sb.style.display='inline-block';
        if(model==='karaoke'){
          buildKaraokeView(d.stems,d.filename||pname);
        }else{
          buildMixer(d.stems,d.filename||pname);
          if(d.mixer_state&&Object.keys(d.mixer_state).length)
            setTimeout(function(){pjRestoreMixer(d.mixer_state);},80);
          checkAiStatus();
        }
      })
      .catch(function(e){alert('Error: '+e.message);});
  }

  function pjRestoreMixer(saved){
    Object.keys(saved).forEach(function(n){
      if(!state[n])return;
      var s=saved[n];
      state[n].muted=!!s.muted;state[n].soloed=!!s.soloed;state[n].vol=s.vol!=null?s.vol:100;
      var sl=document.getElementById('v-'+n),lb=document.getElementById('vl-'+n);
      if(sl){sl.value=state[n].vol;fillSlider(sl,state[n].vol);}
      if(lb)lb.textContent=state[n].vol+'%';
    });
    refresh();
  }

  function pjSave(){
    if(!jobId)return;
    var btn=document.getElementById('pjSaveBtn');if(btn)btn.textContent='Saving...';
    var ms={};
    Object.keys(state).forEach(function(n){ms[n]={vol:state[n].vol,muted:state[n].muted,soloed:state[n].soloed};});
    fetch('/api/projects/'+jobId+'/save_state',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mixer_state:ms})})
      .then(function(r){return r.json();})
      .then(function(){if(btn){btn.innerHTML='&#10003; Saved';setTimeout(function(){btn.innerHTML='&#128190; Save';},2000);}pjLoad();})
      .catch(function(){if(btn)btn.innerHTML='&#128190; Save';});
  }

  // Wire event listeners for project panel buttons
  document.getElementById('pjHdr').addEventListener('click',function(e){
    if(!e.target.closest('.pj-hdr-right'))pjToggle();
  });
  document.getElementById('pjSaveBtn').addEventListener('click',function(e){
    e.stopPropagation(); pjSave();
  });

  // Load projects on startup
  pjLoad();

  // ── SETTINGS PANEL ───────────────────────────────────────────────────
  (function() {
    var open = false;

    function cfgToggle() {
      open = !open;
      var body = document.getElementById('cfgBody');
      var chev = document.getElementById('cfgChev');
      if (body) body.className = open ? 'cfg-body open' : 'cfg-body';
      if (chev) chev.className = open ? 'pj-chev open' : 'pj-chev';
    }

    function cfgLoadStatus() {
      fetch('/api/config')
        .then(function(r){return r.json();})
        .then(function(d){
          var gb = document.getElementById('cfgGeniusBadge');
          var hb = document.getElementById('cfgHfBadge');
          if (gb) { gb.textContent = d.has_genius_token ? 'Genius: set' : 'Genius: not set';
                    gb.className   = 'cfg-badge ' + (d.has_genius_token ? 'set' : 'unset'); }
          if (hb) { hb.textContent = d.has_hf_token ? 'HuggingFace: set' : 'HuggingFace: not set';
                    hb.className   = 'cfg-badge ' + (d.has_hf_token ? 'set' : 'unset'); }
        }).catch(function(){});
    }

    function cfgSave() {
      var genius = document.getElementById('cfgGenius').value.trim();
      var hf     = document.getElementById('cfgHf').value.trim();
      var status = document.getElementById('cfgStatus');
      var btn    = document.getElementById('cfgSaveBtn');
      if (!genius && !hf) { if (status) status.textContent = 'Enter at least one key.'; return; }
      if (btn) btn.textContent = 'Saving...';
      var payload = {};
      if (genius) payload.genius_token = genius;
      if (hf)     payload.hf_token     = hf;
      fetch('/api/config', { method:'POST', headers:{'Content-Type':'application/json'},
                             body: JSON.stringify(payload) })
        .then(function(r){return r.json();})
        .then(function(){
          if (status) status.textContent = 'Saved!';
          if (btn) btn.textContent = 'Save Keys';
          document.getElementById('cfgGenius').value = '';
          document.getElementById('cfgHf').value     = '';
          cfgLoadStatus();
          setTimeout(function(){ if(status) status.textContent=''; }, 3000);
        })
        .catch(function(e){ if(status) status.textContent = 'Error: '+e.message; });
    }

    var hdr = document.getElementById('cfgHdr');
    var sav = document.getElementById('cfgSaveBtn');
    if (hdr) hdr.addEventListener('click', cfgToggle);
    if (sav) sav.addEventListener('click', function(e){ e.stopPropagation(); cfgSave(); });
    cfgLoadStatus();
  })();


  // ── LYRICS TEST BUTTON ───────────────────────────────────────────────
  (function() {
    var tb = document.getElementById('kTestBtn');
    if (!tb) return;
    tb.addEventListener('click', function() {
      var titleEl = document.getElementById('kSongTitle');
      var query   = titleEl ? titleEl.value.trim() : '';
      tb.textContent = 'Testing...'; tb.disabled = true;
      fetch('/api/test_lyrics', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({query: query})
      })
      .then(function(r){ return r.json(); })
      .then(function(d) {
        tb.textContent = 'Test'; tb.disabled = false;
        var lines = [];
        if (!d.genius_token_set)  lines.push('No Genius token saved in AI Settings.');
        if (d.lyricsgenius_installed === false) lines.push('lyricsgenius not installed. Run: pip install lyricsgenius');
        if (d.error)              lines.push('ERROR: ' + d.error);
        if (d.song_found)         lines.push('Found: ' + d.song_found + ' by ' + d.artist_found);
        if (d.attributed_singers && d.attributed_singers.length)
          lines.push('Singers in lyrics: ' + d.attributed_singers.join(', '));
        if (d.warning)            lines.push('Warning: ' + d.warning);
        if (d.song_found && d.has_attribution)
          lines.push('Ready! Click Auto-detect Singers.');
        alert(lines.join('\\n') || JSON.stringify(d));
      })
      .catch(function(e) {
        tb.textContent = 'Test'; tb.disabled = false;
        alert('Test failed: ' + e.message);
      });
    });
  })();

  function err(msg){
    clearErr();
    var box=document.createElement('div');box.className='err-box';
    box.innerHTML='<strong>Something went wrong</strong>'+msg;
    document.querySelector('main').insertBefore(box,document.getElementById('view-upload'));
    show('upload');
  }
  function clearErr(){document.querySelectorAll('.err-box').forEach(function(e){e.remove();});}
})();
</script>
</body>
</html>"""


@app.route('/')
def index(): return HTML

@app.route('/api/separate',methods=['POST'])
def separate():
    if 'file' not in request.files: return jsonify({'error':'No file'}),400
    f=request.files['file'];mdl=request.form.get('model','htdemucs')
    jid=str(uuid.uuid4())[:10];ext=os.path.splitext(f.filename)[1] if f.filename else '.mp3'
    up=os.path.join(UPLOAD_DIR,jid+ext);f.save(up)
    # Return job_id immediately — duration detection + separation run in background
    mode_label='karaoke (vocals + instrumental)' if mdl=='karaoke' else f'{mdl} stems'
    jobs[jid]={'status':'processing','message':f'Starting {mode_label} separation...','stems':[],
               'filename':f.filename or 'audio','upload_path':os.path.abspath(up),'duration':30.0,'progress':0,'pass_num':1}
    t=threading.Thread(target=run_demucs,args=(jid,up,mdl));t.daemon=True;t.start()
    return jsonify({'job_id':jid})

def run_demucs(jid,up,mdl):
    try:
        # Detect duration in background (non-blocking for route)
        try:
            from pydub import AudioSegment as _AS
            jobs[jid]['duration']=round(len(_AS.from_file(up))/1000.0,1)
        except Exception:
            pass

        jobs[jid]['progress']=0
        odir=os.path.join(OUTPUT_DIR,jid);os.makedirs(odir,exist_ok=True)
        if mdl=='karaoke':
            jobs[jid]['message']='Loading karaoke model (may download ~600MB on first run)...'
            cmd=[sys.executable,'-m','demucs','-n','htdemucs_ft','--two-stems','vocals','--mp3','-o',odir,up]
        else:
            cmd=[sys.executable,'-m','demucs','-n',mdl,'--mp3','-o',odir,up]

        # Force UTF-8 in the subprocess so non-ASCII filenames don't crash on Windows
        _env = os.environ.copy()
        _env['PYTHONIOENCODING'] = 'utf-8'
        _env['PYTHONUTF8']       = '1'
        proc=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_env)
        stderr_buf=[]

        _pass=[1]; _last_pct=[0]
        def read_err():
            buf=b''
            while True:
                ch=proc.stderr.read(1)
                if not ch: break
                if ch in (b'\r',b'\n'):
                    if buf:
                        line=buf.decode('utf-8',errors='replace').strip()
                        if line:
                            stderr_buf.append(line)
                            # Parse tqdm percentage
                            m=re.search(r'(\d+)%',line)
                            if m:
                                pct=int(m.group(1))
                                if pct < _last_pct[0] - 15:
                                    _pass[0]+=1
                                _last_pct[0]=pct
                                jobs[jid]['progress']=pct
                                jobs[jid]['pass_num']=_pass[0]
                                jobs[jid]['message']=f'Step {_pass[0]}: separating...' if pct<100 else f'Step {_pass[0]} done, continuing...'
                            else:
                                # Show any other output (downloads, model loading, etc.)
                                # Filter out noisy debug lines
                                if not any(skip in line.lower() for skip in ['warning','deprecated','userwarning','futurewarning']):
                                    short=line[:80]
                                    if short:
                                        jobs[jid]['message']=short
                    buf=b''
                else:
                    buf+=ch

        et=threading.Thread(target=read_err,daemon=True); et.start()
        proc.wait(); et.join(timeout=5)

        if proc.returncode!=0:
            err_text='\n'.join(stderr_buf)
            jobs[jid]['status']='error'
            jobs[jid]['message']=(err_text or 'Unknown error')[-600:]
            return

        files=glob.glob(os.path.join(odir,'**','*.mp3'),recursive=True)
        jobs[jid]['stems']=[{'name':os.path.splitext(os.path.basename(p))[0],'path':os.path.abspath(p)} for p in sorted(files)]
        jobs[jid]['progress']=100
        jobs[jid]['status']='done'
        jobs[jid]['message']='Done!'
        jobs[jid]['_pj_model']=mdl
        _pj_save(jid)
    except Exception as e:
        jobs[jid]['status']='error';jobs[jid]['message']=str(e)

@app.route('/api/status/<jid>')
def status(jid):
    job=jobs.get(jid)
    if not job: return jsonify({'status':'not_found'}),404
    out={'status':job['status'],'message':job['message'],'filename':job.get('filename',''),
         'duration':job.get('duration',30.0),'cleaned':job.get('cleaned',False),
         'progress':job.get('progress',0),'pass_num':job.get('pass_num',1)}
    if job['status']=='done':
        out['stems']=[{'name':s['name'],'download_url':f'/api/dl/{jid}/{s["name"]}'} for s in job['stems']]
    return jsonify(out)

@app.route('/api/dl/<jid>/<stem>')
def dl_stem(jid,stem):
    job=jobs.get(jid)
    if not job or job['status']!='done': return jsonify({'error':'Not ready'}),404
    for s in job['stems']:
        if s['name']==stem: return send_file(s['path'],as_attachment=True,download_name=stem+'.mp3',mimetype='audio/mpeg')
    return jsonify({'error':'Not found'}),404

@app.route('/api/original/<jid>')
def original(jid):
    job=jobs.get(jid)
    if not job: return jsonify({'error':'Not found'}),404
    p=job.get('upload_path','')
    if not p or not os.path.exists(p): return jsonify({'error':'File missing'}),404
    return send_file(p,mimetype='audio/mpeg')

@app.route('/api/mix/<jid>',methods=['POST'])
def mix(jid):
    job=jobs.get(jid)
    if not job or job['status']!='done': return jsonify({'error':'Job not ready'}),404
    try:
        from pydub import AudioSegment
    except ImportError as e:
        return jsonify({'error':f'pydub import failed: {e}. Run: pip install pydub'}),500
    except Exception as e:
        return jsonify({'error':f'pydub load error: {e}'}),500
    try:
        data=request.get_json();cfg=data.get('stems',{});fmt=data.get('format','mp3')
        if fmt not in ('mp3','wav','flac'): fmt='mp3'
        segs=[]
        for s in job['stems']:
            n=s['name'];c=cfg.get(n,{})
            if c.get('muted'): continue
            vol=c.get('volume',100)
            if vol==0: continue
            # Use AI replacement if available and requested
            stem_path = s['path']
            if c.get('use_ai_replacement'):
                rep_map = job.get('stem_replacements', {})
                rep_path = rep_map.get(n, '')
                if rep_path and os.path.exists(rep_path):
                    stem_path = rep_path
            if not os.path.exists(stem_path):
                return jsonify({'error':f'Stem file missing: {stem_path}. Try re-separating the track.'}),500
            audio=AudioSegment.from_mp3(stem_path)
            if vol!=100: audio=audio+(20.0*math.log10(max(vol,1)/100.0))
            segs.append(audio)
        if not segs: return jsonify({'error':'All stems muted'}),400
        mixed=segs[0]
        for seg in segs[1:]: mixed=mixed.overlay(seg)
        out=os.path.join(OUTPUT_DIR,jid,'mix.'+fmt)
        mixed.export(out,format=fmt)
        mime={'mp3':'audio/mpeg','wav':'audio/wav','flac':'audio/flac'}.get(fmt,'audio/mpeg')
        return send_file(out,as_attachment=True,download_name='mix.'+fmt,mimetype=mime)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error':f'Mix failed: {e}'}),500




# ── VOCAL SPLIT (lead vs backing) ────────────────────────────────────
def run_vocal_split(jid, vocals_path, split_jid):
    try:
        try:
            from audio_separator.separator import Separator
        except ImportError as _ie:
            _err = str(_ie)
            if 'onnxruntime' in _err:
                _fix = 'pip install "audio-separator[cpu]" onnxruntime'
            else:
                _fix = 'pip install "audio-separator[cpu]" onnxruntime'
            split_jobs[split_jid]['status']  = 'error'
            split_jobs[split_jid]['message'] = f'Import failed ({_err}). Try: {_fix}'
            return
        except Exception as _ie:
            split_jobs[split_jid]['status']  = 'error'
            split_jobs[split_jid]['message'] = f'audio-separator load error: {_ie}'
            return

        job = jobs.get(jid)
        if not job:
            split_jobs[split_jid]['status']  = 'error'
            split_jobs[split_jid]['message'] = 'Source job not found'
            return

        out_dir = os.path.abspath(os.path.join(OUTPUT_DIR, jid, 'vocal_split'))
        os.makedirs(out_dir, exist_ok=True)

        split_jobs[split_jid]['message'] = 'Loading model (downloading ~50 MB on first run)...'

        # audio-separator puts files in output_dir; we also record cwd to catch strays
        cwd_before = os.getcwd()

        # Try multiple constructor signatures across audio-separator versions
        sep = None
        for kwargs in [
            {'output_format': 'wav', 'output_dir': out_dir, 'log_level': 30},
            {'output_format': 'wav', 'output_dir': out_dir},
            {'output_dir': out_dir},
            {},
        ]:
            try:
                sep = Separator(**kwargs)
                break
            except TypeError:
                continue
        if sep is None:
            sep = Separator()

        sep.load_model(model_filename='UVR_MDXNET_KARA_2.onnx')
        split_jobs[split_jid]['message'] = 'Separating lead and backing vocals...'

        result = sep.separate(vocals_path)
        print(f'[VocalSplit] result={result}')
        print(f'[VocalSplit] out_dir contents: {os.listdir(out_dir)}')

        # ── Collect all candidate audio files ────────────────────────
        AUDIO_EXTS = ('*.wav', '*.mp3', '*.flac', '*.m4a')
        candidates = []
        for ext in AUDIO_EXTS:
            candidates += glob.glob(os.path.join(out_dir, ext))
            candidates += glob.glob(os.path.join(cwd_before, ext.replace('*', '*_(*)_*')))

        # Also use paths returned directly by result
        if result:
            for r in (result if isinstance(result, (list, tuple)) else [result]):
                rp = str(r)
                if os.path.exists(rp):
                    candidates.append(rp)
                # Try relative to cwd
                abs_r = os.path.join(cwd_before, os.path.basename(rp))
                if os.path.exists(abs_r):
                    candidates.append(abs_r)

        candidates = list({os.path.abspath(c) for c in candidates if os.path.exists(c)})
        print(f'[VocalSplit] candidates: {candidates}')

        if not candidates:
            split_jobs[split_jid]['status']  = 'error'
            split_jobs[split_jid]['message'] = (
                f'No output files found. out_dir={out_dir}, '
                f'cwd={cwd_before}, result={result}')
            return

        # ── Identify lead vs backing ─────────────────────────────────
        lead_path    = None
        backing_path = None

        # Priority: use result order if we have 2+ files from result
        result_paths = []
        if result:
            for r in (result if isinstance(result, (list, tuple)) else [result]):
                rp = str(r)
                if not os.path.exists(rp):
                    rp = os.path.join(cwd_before, os.path.basename(rp))
                if os.path.exists(rp):
                    result_paths.append(os.path.abspath(rp))

        if len(result_paths) >= 2:
            p_lo = result_paths[0].lower()
            s_lo = result_paths[1].lower()
            if any(k in p_lo for k in ('instrumental','backing','no_vocal','_(no')):
                lead_path, backing_path = result_paths[1], result_paths[0]
            else:
                lead_path, backing_path = result_paths[0], result_paths[1]
        else:
            # Sort by filename and classify by keywords
            for f in sorted(candidates):
                fl = os.path.basename(f).lower()
                is_backing = any(k in fl for k in
                    ('instrumental','backing','no_vocal','_(no','karaoke','accomp'))
                if is_backing:
                    if backing_path is None: backing_path = f
                else:
                    if lead_path is None: lead_path = f

            # Last resort: just use first two files
            if not lead_path and candidates:
                lead_path = candidates[0]
            if not backing_path and len(candidates) > 1:
                backing_path = candidates[1]

        if not lead_path:
            split_jobs[split_jid]['status']  = 'error'
            split_jobs[split_jid]['message'] = (
                f'Files found but could not classify them: {candidates}')
            return

        if not backing_path:
            backing_path = lead_path

        # ── Convert to MP3 if needed (pydub) ─────────────────────────
        def ensure_mp3(src, dst_name):
            if src.lower().endswith('.mp3'):
                return src
            try:
                from pydub import AudioSegment
                dst = os.path.join(out_dir, dst_name)
                AudioSegment.from_file(src).export(dst, format='mp3', bitrate='320k')
                return dst
            except Exception:
                return src  # use original if conversion fails

        lead_path    = ensure_mp3(lead_path,    'lead_vocals.mp3')
        backing_path = ensure_mp3(backing_path, 'backing_vocals.mp3')

        lead_path    = os.path.abspath(lead_path)
        backing_path = os.path.abspath(backing_path)

        # ── Update job stems ─────────────────────────────────────────
        new_stems = [s for s in job['stems'] if s['name'] != 'vocals']
        new_stems.insert(0, {'name': 'lead_vocals',    'path': lead_path})
        new_stems.insert(1, {'name': 'backing_vocals', 'path': backing_path})
        job['stems']        = new_stems
        job['vocals_split'] = True

        split_jobs[split_jid]['status']       = 'done'
        split_jobs[split_jid]['message']      = 'Done! Mixer updated with Lead and Backing stems.'
        split_jobs[split_jid]['lead_path']    = lead_path
        split_jobs[split_jid]['backing_path'] = backing_path

    except Exception as e:
        import traceback; traceback.print_exc()
        split_jobs[split_jid]['status']  = 'error'
        split_jobs[split_jid]['message'] = str(e)


@app.route('/api/split_vocals/<jid>', methods=['POST'])
def split_vocals_route(jid):
    job = jobs.get(jid)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Separation job not ready'}), 404
    if job.get('vocals_split'):
        return jsonify({'error': 'Vocals already split into lead and backing'}), 400

    # Find the vocals stem path
    vocals_path = next((s['path'] for s in job['stems'] if s['name'] == 'vocals'), None)
    if not vocals_path or not os.path.exists(vocals_path):
        return jsonify({'error': 'No vocals stem found — separate in 4 or 6 stem mode first'}), 400

    sjid = str(uuid.uuid4())[:10]
    split_jobs[sjid] = {'status': 'processing', 'message': 'Starting...'}
    t = threading.Thread(target=run_vocal_split, args=(jid, vocals_path, sjid))
    t.daemon = True; t.start()
    return jsonify({'split_job_id': sjid})


@app.route('/api/vocal_split_status/<sjid>')
def vocal_split_status(sjid):
    job = split_jobs.get(sjid)
    if not job: return jsonify({'status': 'not_found'}), 404
    return jsonify({'status': job['status'], 'message': job['message']})

# ── STEM REPLACEMENT ─────────────────────────────────────────────────
def run_replacement(replace_jid, jid, stem_name, prompt):
    try:
        from pydub import AudioSegment
        replace_jobs[replace_jid]['message'] = 'Mixing context stems...'
        job = jobs.get(jid)
        if not job or job['status'] != 'done':
            replace_jobs[replace_jid]['status']  = 'error'
            replace_jobs[replace_jid]['message'] = 'Source job not ready'
            return
        ctx_segs = []
        for s in job['stems']:
            if s['name'] != stem_name and os.path.exists(s['path']):
                try: ctx_segs.append(AudioSegment.from_mp3(s['path']))
                except Exception: pass
        if not ctx_segs:
            replace_jobs[replace_jid]['status']  = 'error'
            replace_jobs[replace_jid]['message'] = 'No other stems available for context'
            return
        ctx_mix = ctx_segs[0]
        for seg in ctx_segs[1:]: ctx_mix = ctx_mix.overlay(seg)
        ctx_dir  = os.path.join(OUTPUT_DIR, jid, 'replacements')
        os.makedirs(ctx_dir, exist_ok=True)
        ctx_path = os.path.join(ctx_dir, f'{stem_name}_context.mp3')
        ctx_mix.export(ctx_path, format='mp3')
        duration = len(ctx_mix) / 1000.0
        replace_jobs[replace_jid]['message'] = 'Sending to AceStep AI...'
        worker_data, code = w_post('/replace', {
            'context_audio_path': os.path.abspath(ctx_path),
            'prompt': prompt, 'duration': duration, 'stem_name': stem_name,
        })
        if 'error' in worker_data or code >= 400:
            replace_jobs[replace_jid]['status']  = 'error'
            replace_jobs[replace_jid]['message'] = worker_data.get('error', 'Worker error')
            return
        worker_jid = worker_data['job_id']
        import time
        for _ in range(240):
            time.sleep(3)
            st, _ = w_get(f'/job/{worker_jid}')
            replace_jobs[replace_jid]['message'] = st.get('message', 'Generating...')
            if   st.get('status') == 'done':  break
            elif st.get('status') == 'error':
                replace_jobs[replace_jid]['status']  = 'error'
                replace_jobs[replace_jid]['message'] = st.get('message', 'Generation failed')
                return
        else:
            replace_jobs[replace_jid]['status']  = 'error'
            replace_jobs[replace_jid]['message'] = 'Timed out'
            return
        try:
            with urllib.request.urlopen(f'{WORKER}/download/{worker_jid}', timeout=30) as r:
                audio_bytes = r.read()
        except Exception as e:
            replace_jobs[replace_jid]['status']  = 'error'
            replace_jobs[replace_jid]['message'] = f'Failed to retrieve result: {e}'
            return
        result_path = os.path.join(ctx_dir, f'{stem_name}_ai.mp3')
        with open(result_path, 'wb') as f: f.write(audio_bytes)
        if 'stem_replacements' not in job: job['stem_replacements'] = {}
        job['stem_replacements'][stem_name] = os.path.abspath(result_path)
        replace_jobs[replace_jid]['status']  = 'done'
        replace_jobs[replace_jid]['message'] = 'Replacement ready!'
        replace_jobs[replace_jid]['path']    = os.path.abspath(result_path)
    except Exception as e:
        import traceback; traceback.print_exc()
        replace_jobs[replace_jid]['status']  = 'error'
        replace_jobs[replace_jid]['message'] = str(e)


@app.route('/api/replace_stem/<jid>/<stem_name>', methods=['POST'])
def replace_stem_route(jid, stem_name):
    job = jobs.get(jid)
    if not job or job['status'] != 'done': return jsonify({'error':'Separation not ready'}), 404
    data   = request.get_json() or {}
    prompt = data.get('prompt','').strip()
    if not prompt: return jsonify({'error':'Prompt required'}), 400
    rjid = str(uuid.uuid4())[:10]
    replace_jobs[rjid] = {'status':'processing','message':'Starting...','jid':jid,'stem_name':stem_name,'path':None}
    t = threading.Thread(target=run_replacement, args=(rjid, jid, stem_name, prompt))
    t.daemon = True; t.start()
    return jsonify({'replace_job_id': rjid})


@app.route('/api/replace_status/<rjid>')
def replace_status(rjid):
    job = replace_jobs.get(rjid)
    if not job: return jsonify({'status':'not_found'}), 404
    return jsonify({'status':job['status'],'message':job['message']})

# ── KARAOKE ROUTES ───────────────────────────────────────────────────
def silence_region_pydub(audio, start_ms, end_ms, fade_ms=80):
    """Replace a time region with silence, applying smooth crossfades at edges."""
    from pydub import AudioSegment
    region_dur = end_ms - start_ms
    if region_dur <= 0:
        return audio
    fade = min(fade_ms, region_dur // 4, 100)
    before = audio[:start_ms]
    after  = audio[end_ms:]
    silence = AudioSegment.silent(duration=region_dur, frame_rate=audio.frame_rate)
    if fade > 0:
        if len(before) >= fade:
            before = before[:-fade] + before[-fade:].fade_out(fade)
        if len(after) >= fade:
            after = after[:fade].fade_in(fade) + after[fade:]
    return before + silence + after


@app.route('/api/stem_audio/<jid>/<stem_name>')
def stem_audio(jid, stem_name):
    """Serve a stem MP3 inline (no attachment) for WaveSurfer to load."""
    job = jobs.get(jid)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Not ready'}), 404
    for s in job['stems']:
        if s['name'] == stem_name:
            return send_file(s['path'], mimetype='audio/mpeg')
    return jsonify({'error': 'Stem not found'}), 404


@app.route('/api/karaoke_export/<jid>', methods=['POST'])
def karaoke_export(jid):
    job = jobs.get(jid)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Job not ready'}), 404
    try:
        from pydub import AudioSegment
    except ImportError as e:
        return jsonify({'error': f'pydub failed: {e}. Run: pip install pydub audioop-lts'}), 500

    data          = request.get_json() or {}
    regions       = data.get('regions', [])
    silence_label = data.get('silence_label', 'Singer B')
    fmt           = data.get('format', 'mp3')
    if fmt not in ('mp3', 'wav', 'flac'):
        fmt = 'mp3'

    stems_map     = {s['name']: s['path'] for s in job['stems']}
    vocals_path   = stems_map.get('vocals')
    instr_path    = stems_map.get('no_vocals')

    if not vocals_path or not instr_path:
        return jsonify({'error': 'Vocals and instrumental stems not found. Re-separate in Karaoke mode.'}), 400

    try:
        vocals       = AudioSegment.from_mp3(vocals_path)
        instrumental = AudioSegment.from_mp3(instr_path)

        # Process regions highest-start-first so indices stay valid
        to_silence = sorted(
            [r for r in regions if r.get('label') == silence_label],
            key=lambda r: float(r.get('start', 0)),
            reverse=True
        )

        for region in to_silence:
            start_ms = max(0, int(float(region['start']) * 1000))
            end_ms   = min(len(vocals), int(float(region['end']) * 1000))
            if end_ms > start_ms:
                vocals = silence_region_pydub(vocals, start_ms, end_ms)

        mixed    = vocals.overlay(instrumental)
        out_path = os.path.join(OUTPUT_DIR, jid, f'karaoke.{fmt}')
        mixed.export(out_path, format=fmt)

        mime = {'mp3':'audio/mpeg','wav':'audio/wav','flac':'audio/flac'}.get(fmt,'audio/mpeg')
        return send_file(out_path, as_attachment=True,
                         download_name=f'karaoke.{fmt}', mimetype=mime)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── YOUTUBE DOWNLOADER ───────────────────────────────────────────────
def run_yt_download(jid, url, browser=''):
    try:
        import yt_dlp
    except ImportError:
        yt_jobs[jid]['status']  = 'error'
        yt_jobs[jid]['message'] = 'yt-dlp not installed. Run: pip install yt-dlp'
        return

    out_dir = os.path.join(YT_DIR, jid)
    os.makedirs(out_dir, exist_ok=True)

    def progress_hook(d):
        if d['status'] == 'downloading':
            dl   = d.get('downloaded_bytes') or 0
            tot  = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
            if tot > 0:
                pct = int(dl / tot * 100)
                mb_dl  = dl  / (1024 * 1024)
                mb_tot = tot / (1024 * 1024)
                yt_jobs[jid]['progress'] = pct
                yt_jobs[jid]['message']  = f'Downloading... {mb_dl:.1f} / {mb_tot:.1f} MB'
            else:
                yt_jobs[jid]['message'] = 'Downloading...'
        elif d['status'] == 'finished':
            yt_jobs[jid]['progress'] = 95
            yt_jobs[jid]['message']  = 'Converting to MP3...'

    try:
        ydl_opts = {
            'format':           'bestaudio/best',
            'postprocessors':   [{'key': 'FFmpegExtractAudio',
                                  'preferredcodec': 'mp3',
                                  'preferredquality': '320'}],
            'outtmpl':          os.path.join(out_dir, '%(title)s.%(ext)s'),
            'progress_hooks':   [progress_hook],
            'quiet':            True,
            'no_warnings':      True,
            'noplaylist':       True,
            'restrictfilenames': True,
            'windowsfilenames':  True,
        }
        # Use browser cookies for private/login-required content (e.g. YouTube Music)
        if browser:
            ydl_opts['cookiesfrombrowser'] = (browser,)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info  = ydl.extract_info(url, download=True)
            title = info.get('title', 'audio') if info else 'audio'

        mp3_files = glob.glob(os.path.join(out_dir, '*.mp3'))
        if mp3_files:
            mp3_path = os.path.abspath(mp3_files[0])
            yt_jobs[jid]['status']   = 'done'
            yt_jobs[jid]['path']     = mp3_path
            yt_jobs[jid]['filename'] = os.path.basename(mp3_path)
            yt_jobs[jid]['title']    = title
            yt_jobs[jid]['progress'] = 100
            yt_jobs[jid]['message']  = 'Complete!'
        else:
            yt_jobs[jid]['status']  = 'error'
            yt_jobs[jid]['message'] = 'MP3 not found after conversion. FFmpeg may be missing.'
    except Exception as e:
        msg = re.sub(r'\x1b\[[0-9;]*m', '', str(e))
        auth_kw = ['sign in', 'login', 'not a bot', '403', 'private video',
                   'members only', 'premium', 'unavailable', 'age']
        if any(k in msg.lower() for k in auth_kw):
            msg = ('Login required. Select the browser where you are logged in to '
                   'YouTube in the "Browser cookies" selector, then try again.')
        yt_jobs[jid]['status']  = 'error'
        yt_jobs[jid]['message'] = msg


@app.route('/api/yt/download', methods=['POST'])
def yt_download():
    data = request.get_json() or {}
    url  = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    if 'youtube.com' not in url and 'youtu.be' not in url:
        return jsonify({'error': 'Please provide a valid YouTube URL'}), 400
    jid = str(uuid.uuid4())[:10]
    browser = data.get('browser', '').strip().lower()
    yt_jobs[jid] = {'status':'processing','message':'Connecting to YouTube...','progress':0,
                    'path':None,'filename':'','title':''}
    t = threading.Thread(target=run_yt_download, args=(jid, url, browser))
    t.daemon = True; t.start()
    return jsonify({'job_id': jid})


@app.route('/api/yt/status/<jid>')
def yt_status(jid):
    job = yt_jobs.get(jid)
    if not job: return jsonify({'status':'not_found'}), 404
    return jsonify({
        'status':   job['status'],
        'message':  job['message'],
        'progress': job['progress'],
        'title':    job.get('title',''),
        'filename': job.get('filename',''),
    })


@app.route('/api/yt/file/<jid>')
def yt_file(jid):
    job = yt_jobs.get(jid)
    if not job or job['status'] != 'done': return jsonify({'error':'Not ready'}), 404
    p = job.get('path','')
    if not p or not os.path.exists(p):    return jsonify({'error':'File missing'}), 404
    return send_file(p, as_attachment=True,
                     download_name=job.get('filename','audio.mp3'),
                     mimetype='audio/mpeg')


@app.route('/api/yt/open_login', methods=['POST'])
def yt_open_login():
    import webbrowser
    webbrowser.open('https://music.youtube.com')
    return jsonify({'ok': True})


@app.route('/api/yt/separate/<yt_jid>', methods=['POST'])
def yt_separate(yt_jid):
    yt_job = yt_jobs.get(yt_jid)
    if not yt_job or yt_job['status'] != 'done':
        return jsonify({'error': 'YouTube download not ready'}), 404
    yt_path = yt_job.get('path','')
    if not yt_path or not os.path.exists(yt_path):
        return jsonify({'error': 'File not found on disk'}), 404
    data  = request.get_json() or {}
    mdl   = data.get('model', 'htdemucs')
    fname = yt_job.get('filename', 'audio.mp3')
    jid   = str(uuid.uuid4())[:10]
    # Detect duration in bg - don't block
    jobs[jid] = {
        'status': 'processing', 'message': 'Starting separation...',
        'stems': [], 'filename': fname,
        'upload_path': yt_path, 'duration': 30.0,
        'progress': 0, 'pass_num': 1
    }
    t = threading.Thread(target=run_demucs, args=(jid, yt_path, mdl))
    t.daemon = True; t.start()
    return jsonify({'job_id': jid})

# ── PROJECT MANAGEMENT ────────────────────────────────────────────────
import json as _pj_json
from datetime import datetime as _pj_dt, timezone as _pj_tz

_PROJECTS_DIR = 'projects'
os.makedirs(_PROJECTS_DIR, exist_ok=True)
_PJ_BASE = os.path.dirname(os.path.abspath(__file__))

def _pj_rel(p):
    if not p: return ''
    try:    return os.path.relpath(p, _PJ_BASE)
    except: return p

def _pj_abs(p):
    if not p: return ''
    if os.path.isabs(p): return p
    return os.path.join(_PJ_BASE, p)

def _pj_save(jid):
    try:
        job = jobs.get(jid)
        if not job or job['status'] != 'done': return
        proj = {
            'id':          jid,
            'name':        job.get('filename','Untitled').rsplit('.',1)[0],
            'created':     job.get('_pj_ts', _pj_dt.now(_pj_tz.utc).isoformat()),
            'modified':    _pj_dt.now(_pj_tz.utc).isoformat(),
            'source_file': job.get('filename',''),
            'source_path': _pj_rel(job.get('upload_path','')),
            'duration':    job.get('duration', 0),
            'model':       job.get('_pj_model','htdemucs'),
            'stems':       [{'name':s['name'],'path':_pj_rel(s['path'])} for s in job.get('stems',[])],
            'cleaned':         job.get('cleaned', False),
            'vocals_split':    job.get('vocals_split', False),
            'stem_replacements': {k:_pj_rel(v) for k,v in job.get('stem_replacements',{}).items()},
            'mixer_state':     job.get('mixer_state', {}),
        }
        pd = os.path.join(_PROJECTS_DIR, jid)
        os.makedirs(pd, exist_ok=True)
        with open(os.path.join(pd,'project.json'),'w') as f:
            _pj_json.dump(proj, f, indent=2)
    except Exception as e:
        print(f'[Project] save failed: {e}')

def _pj_load(pid):
    pf = os.path.join(_PROJECTS_DIR, pid, 'project.json')
    if not os.path.exists(pf): return None
    with open(pf) as f: proj = _pj_json.load(f)
    proj['stems_abs'] = [
        {'name':s['name'],'path':_pj_abs(s['path'])}
        for s in proj.get('stems',[])
        if _pj_abs(s['path']) and os.path.exists(_pj_abs(s['path']))
    ]
    proj['replacements_abs'] = {
        k: _pj_abs(v) for k,v in proj.get('stem_replacements',{}).items()
        if _pj_abs(v) and os.path.exists(_pj_abs(v))
    }
    return proj

@app.route('/api/projects')
def pj_list():
    result = []
    if os.path.exists(_PROJECTS_DIR):
        for pid in os.listdir(_PROJECTS_DIR):
            pf = os.path.join(_PROJECTS_DIR, pid, 'project.json')
            if not os.path.exists(pf): continue
            try:
                with open(pf) as f: proj = _pj_json.load(f)
                if any(os.path.exists(_pj_abs(s['path'])) for s in proj.get('stems',[])):
                    result.append(proj)
            except Exception: pass
    result.sort(key=lambda p: p.get('modified',''), reverse=True)
    return jsonify(result)

@app.route('/api/projects/<pid>/load')
def pj_load_route(pid):
    proj = _pj_load(pid)
    if not proj:               return jsonify({'error':'Not found'}), 404
    if not proj['stems_abs']:  return jsonify({'error':'Audio files missing from disk'}), 404
    jobs[pid] = {
        'status':'done','message':'Loaded','progress':100,'pass_num':1,
        'stems':   proj['stems_abs'],
        'filename':proj.get('source_file',''),
        'upload_path': _pj_abs(proj.get('source_path','')),
        'duration':    proj.get('duration', 30.0),
        '_pj_model':   proj.get('model','htdemucs'),
        'model':       proj.get('model','htdemucs'),
        'cleaned':     proj.get('cleaned', False),
        'vocals_split':proj.get('vocals_split', False),
        'stem_replacements': proj.get('replacements_abs',{}),
        '_pj_ts':      proj.get('created',''),
    }
    return jsonify({
        'job_id':    pid,
        'stems':     [{'name':s['name']} for s in proj['stems_abs']],
        'filename':  proj.get('source_file',''),
        'duration':  proj.get('duration', 30.0),
        'model':     proj.get('model','htdemucs'),
        'cleaned':   proj.get('cleaned', False),
        'vocals_split': proj.get('vocals_split', False),
        'mixer_state':  proj.get('mixer_state', {}),
    })

@app.route('/api/projects/<pid>', methods=['DELETE'])
def pj_delete(pid):
    import shutil
    pd = os.path.join(_PROJECTS_DIR, pid)
    if os.path.exists(pd): shutil.rmtree(pd)
    return jsonify({'ok': True})

@app.route('/api/projects/<pid>/save_state', methods=['POST'])
def pj_save_state(pid):
    data = request.get_json() or {}
    pf   = os.path.join(_PROJECTS_DIR, pid, 'project.json')
    if not os.path.exists(pf):
        job = jobs.get(pid)
        if job and job.get('status') == 'done': _pj_save(pid)
        else: return jsonify({'error':'Not found'}), 404
    with open(pf) as f: proj = _pj_json.load(f)
    if 'mixer_state' in data: proj['mixer_state'] = data['mixer_state']
    proj['modified'] = _pj_dt.now(_pj_tz.utc).isoformat()
    with open(pf,'w') as f: _pj_json.dump(proj, f, indent=2)
    return jsonify({'ok': True})

# ── AUTO VOCAL CLASSIFICATION ─────────────────────────────────────────
auto_classify_jobs = {}

def run_auto_classify(jid, vocals_path, labeled_regions, ajid):
    """
    Three-tier classification:
      1. Genius lyrics + Whisper (most accurate for known songs)
      2. pyannote.audio          (best audio-only, needs HF token)
      3. resemblyzer / MFCC      (existing fallback, always available)
    """
    try:
        import numpy as np
        from collections import Counter

        job_meta = jobs.get(jid, {})
        song_hint = job_meta.get('_song_title', '') or job_meta.get('filename', '')

        # ── Attempt 1: Genius lyrics + Whisper ───────────────────────
        genius_token = _load_config().get('genius_token','').strip()
        if genius_token:
            result = _classify_via_lyrics(
                vocals_path, song_hint, genius_token, ajid)
            if result is not None:
                auto_classify_jobs[ajid]['status']   = 'done'
                auto_classify_jobs[ajid]['progress'] = 100
                auto_classify_jobs[ajid]['message']  = (
                    f'Done via Genius lyrics - {len(result)} segments found')
                auto_classify_jobs[ajid]['method']   = 'lyrics'
                auto_classify_jobs[ajid]['regions']  = result
                return

        # ── Attempt 2: pyannote.audio ─────────────────────────────────
        hf_token = _load_config().get('hf_token','').strip()
        if hf_token:
            result = _classify_via_pyannote(
                vocals_path, labeled_regions, hf_token, ajid)
            if result is not None:
                auto_classify_jobs[ajid]['status']   = 'done'
                auto_classify_jobs[ajid]['progress'] = 100
                auto_classify_jobs[ajid]['message']  = (
                    f'Done via pyannote AI - {len(result)} segments found')
                auto_classify_jobs[ajid]['method']   = 'pyannote'
                auto_classify_jobs[ajid]['regions']  = result
                return

        # ── Attempt 3: resemblyzer / MFCC (always available) ─────────
        _classify_via_embeddings(vocals_path, labeled_regions, ajid)

    except Exception as e:
        import traceback; traceback.print_exc()
        auto_classify_jobs[ajid]['status']  = 'error'
        auto_classify_jobs[ajid]['message'] = str(e)


# ── Config helpers ────────────────────────────────────────────────────
_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'stemsplit_config.json')

def _load_config():
    try:
        if os.path.exists(_CONFIG_FILE):
            import json
            with open(_CONFIG_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_config(data):
    import json
    cfg = _load_config()
    cfg.update(data)
    with open(_CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)


# ── Method 1: Genius lyrics + Whisper ────────────────────────────────
def _parse_genius_sections(lyrics_text):
    """Parse Genius lyrics into [{artist, lines}] with singer attribution."""
    import re
    header_re = re.compile(r'\[([^\]]+)\]')
    sections = []
    current_artist = None
    current_lines  = 0
    for line in lyrics_text.split('\n'):
        m = header_re.search(line)
        if m:
            if current_artist and current_lines > 0:
                sections.append({'artist': current_artist, 'lines': current_lines})
            header = m.group(1)
            if ':' in header:
                raw = header.split(':', 1)[1].strip()
                # "Beyoncé & Jay-Z" → take first artist
                first = re.split(r'\s*[&+/,]\s*', raw)[0].strip()
                current_artist = re.sub(r'\([^)]*\)', '', first).strip()
            else:
                current_artist = None   # unattributed section
            current_lines = 0
        elif line.strip() and not line.strip().startswith('['):
            current_lines += 1
    if current_artist and current_lines > 0:
        sections.append({'artist': current_artist, 'lines': current_lines})
    return sections


def _classify_via_lyrics(vocals_path, song_hint, genius_token, ajid):
    """
    Fetch singer-attributed lyrics from Genius, then distribute the
    track duration proportionally across sections.
    No whisper / ML models required — just: pip install lyricsgenius
    """
    import re

    try:
        import lyricsgenius
    except ImportError:
        auto_classify_jobs[ajid]['message'] = (
            'Lyrics method skipped: run  pip install lyricsgenius  then restart')
        return None

    # Get track duration via pydub (always installed)
    try:
        from pydub import AudioSegment
        duration = len(AudioSegment.from_file(vocals_path)) / 1000.0
    except Exception:
        duration = 240.0

    auto_classify_jobs[ajid]['message'] = 'Searching Genius for lyrics...'
    auto_classify_jobs[ajid]['progress'] = 15

    query = re.sub(r'\.(mp3|wav|flac|m4a)$', '', song_hint, flags=re.I)
    query = re.sub(r'[_\-]+', ' ', query).strip()

    if not query:
        auto_classify_jobs[ajid]['message'] = (
            'No song title provided. '
            'Type the artist and song name in the "Song title" field first.')
        return None

    try:
        genius = lyricsgenius.Genius(
            genius_token, verbose=False,
            remove_section_headers=False,
            timeout=15, retries=2)
        genius.skip_non_songs = True
        song = genius.search_song(query)
    except Exception as e:
        auto_classify_jobs[ajid]['message'] = (
            f'Genius search failed: {e}. '
            'Check your Genius API key in the AI Settings panel.')
        return None

    if not song or not song.lyrics:
        auto_classify_jobs[ajid]['message'] = (
            f'"{query}" not found on Genius. '
            'Try entering the exact  Artist - Song Title  in the Song title field above.')
        return None

    auto_classify_jobs[ajid]['progress'] = 50
    auto_classify_jobs[ajid]['message']  = 'Parsing singer attribution from lyrics...'

    sections = _parse_genius_sections(song.lyrics)
    attributed = [s for s in sections if s['artist'] and s['lines'] > 0]
    unique_artists = sorted(set(s['artist'] for s in attributed))

    if len(unique_artists) < 2:
        if unique_artists:
            auto_classify_jobs[ajid]['message'] = (
                f'Lyrics found but only one singer detected ({unique_artists[0]}). '
                'Genius may not have per-singer attribution for this song. '
                'Check the song on genius.com — look for  [Verse 1: Singer Name]  style headers.')
        else:
            auto_classify_jobs[ajid]['message'] = (
                'Lyrics found but no singer headers detected. '
                'Genius may not have per-singer attribution for this song.')
        return None

    auto_classify_jobs[ajid]['progress'] = 70
    auto_classify_jobs[ajid]['message']  = (
        f'Found {len(unique_artists)} singers: '
        + ', '.join(unique_artists[:4])
        + ' — building timeline...')

    # Map artist names to Singer A / B / C labels
    artist_label = {a: f'Singer {chr(65+i)}' for i,a in enumerate(unique_artists[:26])}

    total_lines = sum(s['lines'] for s in attributed)
    regions = []
    t = 0.0

    for s in attributed:
        seg_dur = (s['lines'] / total_lines) * duration
        if seg_dur < 0.5:
            t += seg_dur; continue
        label = artist_label.get(s['artist'], 'Singer A')
        # Merge adjacent same-label sections
        if regions and regions[-1]['label'] == label:
            regions[-1]['end'] = round(t + seg_dur, 2)
        else:
            regions.append({
                'start': round(t, 2),
                'end':   round(t + seg_dur, 2),
                'label': label
            })
        t += seg_dur

    if len(regions) < 2:
        auto_classify_jobs[ajid]['message'] = 'Could not build a timeline from the lyrics.'
        return None

    auto_classify_jobs[ajid]['message'] = (
        f'Done via Genius lyrics ({", ".join(unique_artists[:2])}) '
        f'— {len(regions)} sections. '
        'Note: timing is approximate (proportional to lyric length).')
    return regions



# ── Method 2: pyannote.audio ──────────────────────────────────────────
def _classify_via_pyannote(vocals_path, labeled_regions, hf_token, ajid):
    """Use pyannote speaker diarization. Returns regions or None."""
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        auto_classify_jobs[ajid]['message'] = (
            'pyannote.audio not installed. '
            'Run: pip install pyannote.audio')
        return None

    auto_classify_jobs[ajid]['message'] = 'Loading AI diarization model...'
    auto_classify_jobs[ajid]['progress'] = 10

    try:
        pipeline = Pipeline.from_pretrained(
            'pyannote/speaker-diarization-3.1',
            use_auth_token=hf_token)
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device('cuda'))
    except Exception as e:
        auto_classify_jobs[ajid]['message'] = (
            f'pyannote model load failed: {e}. '
            'Accept the model license at hf.co/pyannote/speaker-diarization-3.1')
        return None

    auto_classify_jobs[ajid]['message'] = 'Running AI speaker diarization...'
    auto_classify_jobs[ajid]['progress'] = 30

    try:
        diarization = pipeline(vocals_path, num_speakers=2)
    except Exception as e:
        auto_classify_jobs[ajid]['message'] = f'Diarization failed: {e}'
        return None

    # Collect turns
    speaker_map = {}
    raw = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_map:
            n = len(speaker_map)
            speaker_map[speaker] = f'Singer {chr(65+n)}'
        raw.append({'start': round(turn.start, 2),
                    'end':   round(turn.end,   2),
                    'label': speaker_map[speaker]})

    if not raw:
        return None

    # If user provided reference regions, remap labels to match
    if labeled_regions:
        raw = _remap_labels(raw, labeled_regions)

    # Merge adjacent same-label, drop short fragments
    return _merge_regions(raw, min_dur=0.5)


# ── Method 3: resemblyzer / MFCC (existing, always available) ────────
def _classify_via_embeddings(vocals_path, labeled_regions, ajid):
    """
    Speaker classification using the best available library.
    Falls back gracefully: resemblyzer → librosa → pydub+numpy (always works).
    """
    import numpy as np
    from collections import Counter

    wav = None; sr = None; using = None

    # ── Try resemblyzer (best quality) ───────────────────────────────
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        auto_classify_jobs[ajid]['message'] = 'Loading voice encoder (resemblyzer)...'
        encoder = VoiceEncoder()
        wav = preprocess_wav(vocals_path)
        sr  = 16000
        def _emb(s, e):
            seg = wav[int(s*sr):int(e*sr)]
            if len(seg) < int(sr*0.5): return None
            try:   return encoder.embed_utterance(seg)
            except: return None
        using = 'Voice Encoder'
    except Exception:
        pass

    # ── Try librosa (good quality) ────────────────────────────────────
    if wav is None:
        try:
            import librosa
            auto_classify_jobs[ajid]['message'] = 'Loading audio (librosa)...'
            wav, sr = librosa.load(vocals_path, sr=22050)
            def _emb(s, e):
                seg = wav[int(s*sr):int(e*sr)]
                if len(seg) < int(sr*0.5): return None
                try:
                    nfft = min(2048, len(seg))
                    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=40, n_fft=nfft)
                    d1   = librosa.feature.delta(mfcc)
                    return np.concatenate([np.mean(mfcc,axis=1), np.std(mfcc,axis=1),
                                           np.mean(d1,axis=1)])
                except: return None
            using = 'MFCC (librosa)'
        except Exception:
            pass

    # ── Fallback: pydub + numpy FFT (always available) ───────────────
    if wav is None:
        try:
            from pydub import AudioSegment
            auto_classify_jobs[ajid]['message'] = 'Loading audio (built-in)...'
            audio = (AudioSegment.from_file(vocals_path)
                     .set_channels(1).set_frame_rate(22050))
            sr  = 22050
            raw = np.array(audio.get_array_of_samples(), dtype=np.float32)
            wav = raw / (np.max(np.abs(raw)) + 1e-8)

            N_FFT = 1024

            def _log_mel_approx(seg, n_bins=40):
                """Log mel-like spectrum using numpy FFT only."""
                n = len(seg)
                spec = np.abs(np.fft.rfft(seg * np.hanning(n), n=N_FFT))
                # Logarithmically-spaced frequency bins (approximates Mel)
                edges = np.logspace(np.log10(1), np.log10(len(spec)-1),
                                    n_bins+1, dtype=int)
                edges = np.clip(edges, 0, len(spec)-1)
                feats = np.array([np.mean(spec[edges[j]:edges[j+1]+1])
                                  for j in range(n_bins)])
                return np.log(feats + 1e-8)

            def _emb(s, e):
                seg = wav[int(s*sr):int(e*sr)]
                n   = len(seg)
                if n < int(sr*0.5): return None
                try:
                    # Average 4 sub-frames to reduce temporal noise
                    sub = n // 4
                    parts = [_log_mel_approx(seg[i*sub:(i+1)*sub])
                             for i in range(4) if sub > N_FFT]
                    if not parts: parts = [_log_mel_approx(seg)]
                    arr = np.array(parts)
                    return np.concatenate([np.mean(arr, axis=0),
                                           np.std(arr,  axis=0)])
                except: return None

            using = 'spectral (built-in)'
        except Exception as e:
            auto_classify_jobs[ajid]['status']  = 'error'
            auto_classify_jobs[ajid]['message'] = (
                f'Could not load audio: {e}. '
                'Try: pip install librosa  or  pip install resemblyzer')
            return

    auto_classify_jobs[ajid]['message'] = f'Analysing vocals ({using})...'

    wav_rms       = float(np.sqrt(np.mean(wav**2)) + 1e-9)
    energy_thresh = wav_rms * 0.06
    duration      = len(wav) / sr

    def _rms(s, e):
        seg = wav[int(s*sr):int(e*sr)]
        return float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0

    # ── Collect embeddings across the track ───────────────────────────
    win_sec = 2.0; step_sec = 0.5
    total   = max(1, int((duration - win_sec) / step_sec) + 1)
    all_embs  = []; all_times = []; done = 0
    t = 0.0
    while t + win_sec <= duration:
        if _rms(t, t+win_sec) >= energy_thresh:
            emb = _emb(t, t+win_sec)
            if emb is not None:
                all_embs.append(emb); all_times.append(t + win_sec/2)
        done += 1
        auto_classify_jobs[ajid]['progress'] = min(50, int(done/total*50))
        t += step_sec

    if len(all_embs) < 4:
        auto_classify_jobs[ajid]['status']  = 'error'
        auto_classify_jobs[ajid]['message'] = 'Not enough voiced audio found.'
        return

    embs_arr = np.array(all_embs)

    # ── Supervised or unsupervised ─────────────────────────────────────
    if labeled_regions:
        ref_embs = {}
        for region in labeled_regions:
            label = region.get('label', 'Singer A')
            emb   = _emb(float(region['start']), float(region['end']))
            if emb is not None:
                ref_embs.setdefault(label, []).append(emb)
        avg_embs = {l: np.mean(e, axis=0) for l,e in ref_embs.items()}

        def _classify(emb):
            sims = {l: float(np.dot(emb,r)/(np.linalg.norm(emb)*np.linalg.norm(r)+1e-8))
                    for l,r in avg_embs.items()}
            sv = sorted(sims.values(), reverse=True)
            return max(sims, key=sims.get), (sv[0]-sv[1] if len(sv)>1 else 1.0)

        lbl_map = None
    else:
        lbl_map = {0: 'Singer A', 1: 'Singer B'}

        def _cosine_kmeans2(X):
            n  = len(X); Xn = X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
            sub = np.linspace(0,n-1,min(n,80),dtype=int); Xs=Xn[sub]
            flat = np.argmin(Xs@Xs.T); i0,i1=divmod(int(flat),len(sub))
            centers = np.array([X[sub[i0]],X[sub[i1]]],dtype=float)
            labels  = np.zeros(n, dtype=int)
            for _ in range(60):
                cn=centers/(np.linalg.norm(centers,axis=1,keepdims=True)+1e-8)
                nl=np.argmax(Xn@cn.T,axis=1)
                if np.array_equal(nl,labels): break
                labels=nl
                for k in range(2):
                    if (labels==k).any(): centers[k]=X[labels==k].mean(0)
            return labels

        cluster_ids = _cosine_kmeans2(embs_arr)
        def _classify(emb): return None, 1.0

    # ── Build frame-level votes ────────────────────────────────────────
    frame_sec   = 0.25
    n_frames    = int(duration/frame_sec)+2
    frame_votes = [[] for _ in range(n_frames)]

    for idx_w,(emb,tc) in enumerate(zip(all_embs,all_times)):
        ts=tc-win_sec/2; te=tc+win_sec/2
        if lbl_map is not None:
            label=lbl_map[int(cluster_ids[idx_w])]; confidence=1.0
        else:
            label,confidence=_classify(emb)
            if confidence<0.02: continue
        for f in range(max(0,int(ts/frame_sec)),min(n_frames,int(te/frame_sec)+1)):
            frame_votes[f].append((label,confidence))

    auto_classify_jobs[ajid]['progress'] = 80
    auto_classify_jobs[ajid]['message']  = 'Smoothing...'

    raw=[]; 
    for v in frame_votes:
        if not v: raw.append(None)
        else:
            sc={}
            for l,c in v: sc[l]=sc.get(l,0.0)+c
            raw.append(max(sc,key=sc.get))

    smoothed=[]
    for i in range(len(raw)):
        wl=[l for l in raw[max(0,i-4):i+5] if l is not None]
        smoothed.append(Counter(wl).most_common(1)[0][0] if wl else None)

    regions=[]; i=0
    while i<len(smoothed):
        if smoothed[i] is None: i+=1; continue
        lbl=smoothed[i]; j=i+1
        while j<len(smoothed) and smoothed[j]==lbl: j+=1
        ss=round(i*frame_sec,2); se=round(min(j*frame_sec,duration),2)
        if se-ss>=0.75: regions.append({'start':ss,'end':se,'label':lbl})
        i=j

    auto_classify_jobs[ajid]['status']   = 'done'
    auto_classify_jobs[ajid]['progress'] = 100
    auto_classify_jobs[ajid]['message']  = f'Done ({using}) - {len(regions)} segments'
    auto_classify_jobs[ajid]['method']   = 'embeddings'
    auto_classify_jobs[ajid]['regions']  = regions


# ── Shared helpers ────────────────────────────────────────────────────
def _merge_regions(regions, min_dur=0.5):
    if not regions: return []
    regions = sorted(regions, key=lambda r: r['start'])
    merged  = [regions[0].copy()]
    for r in regions[1:]:
        if r['label']==merged[-1]['label'] and r['start']<=merged[-1]['end']+0.3:
            merged[-1]['end'] = max(merged[-1]['end'], r['end'])
        else:
            if r['end']-r['start'] >= min_dur:
                merged.append(r.copy())
    return [r for r in merged if r['end']-r['start'] >= min_dur]

def _remap_labels(regions, labeled_regions):
    """Re-map auto-detected labels to match user-provided reference regions."""
    from collections import Counter
    label_overlap = {}
    for region in regions:
        for ref in labeled_regions:
            overlap = min(region['end'],float(ref['end'])) - max(region['start'],float(ref['start']))
            if overlap > 0:
                key = region['label']
                label_overlap.setdefault(key,[]).append(ref['label'])
    remap = {}
    for auto_lbl, ref_lbls in label_overlap.items():
        remap[auto_lbl] = Counter(ref_lbls).most_common(1)[0][0]
    for r in regions:
        r['label'] = remap.get(r['label'], r['label'])
    return regions


# ── Config routes ─────────────────────────────────────────────────────
@app.route('/api/config', methods=['GET', 'POST'])
def config():
    if request.method == 'GET':
        cfg = _load_config()
        return jsonify({
            'has_genius_token': bool(cfg.get('genius_token','')),
            'has_hf_token':     bool(cfg.get('hf_token','')),
        })
    data = request.get_json() or {}
    allowed = {'genius_token', 'hf_token'}
    to_save = {k: v for k, v in data.items() if k in allowed and v}
    if not to_save:
        return jsonify({'error': 'No valid keys provided'}), 400
    _save_config(to_save)
    return jsonify({'ok': True})


@app.route('/api/karaoke_autoclassify/<jid>', methods=['POST'])
def karaoke_autoclassify(jid):
    job = jobs.get(jid)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Job not ready'}), 404
    vocals_path = next(
        (s['path'] for s in job['stems'] if s['name'] in ('vocals', 'lead_vocals')),
        None
    )
    if not vocals_path or not os.path.exists(vocals_path):
        return jsonify({'error': 'No vocals stem found. Separate in 4-stem or karaoke mode first.'}), 400
    data       = request.get_json() or {}
    labeled    = data.get('regions', [])
    song_title = data.get('song_title', '').strip()
    ajid       = str(uuid.uuid4())[:10]
    auto_classify_jobs[ajid] = {
        'status': 'processing', 'message': 'Starting...', 'progress': 0, 'regions': []
    }
    # song_title overrides filename for Genius search
    if song_title:
        jobs[jid]['_song_title'] = song_title
    t = threading.Thread(target=run_auto_classify, args=(jid, vocals_path, labeled, ajid))
    t.daemon = True; t.start()
    return jsonify({'auto_job_id': ajid})


@app.route('/api/karaoke_classify_status/<ajid>')
def karaoke_classify_status(ajid):
    job = auto_classify_jobs.get(ajid)
    if not job: return jsonify({'status': 'not_found'}), 404
    out = {'status': job['status'], 'message': job['message'], 'progress': job['progress']}
    if job['status'] == 'done': out['regions'] = job['regions']
    return jsonify(out)


@app.route('/api/test_lyrics', methods=['POST'])
def test_lyrics_route():
    data  = request.get_json() or {}
    query = data.get('query','').strip()
    out   = {}
    cfg   = _load_config()
    out['genius_token_set'] = bool(cfg.get('genius_token',''))
    out['hf_token_set']     = bool(cfg.get('hf_token',''))
    if not cfg.get('genius_token',''):
        out['error'] = 'No Genius token saved. Open AI Settings and save your token first.'
        return jsonify(out)
    try:
        import lyricsgenius
        out['lyricsgenius_installed'] = True
    except ImportError:
        out['lyricsgenius_installed'] = False
        out['error'] = 'lyricsgenius not installed. Run: pip install lyricsgenius'
        return jsonify(out)
    if not query:
        out['error'] = 'No song title entered. Type Artist - Song Name in the field above.'
        return jsonify(out)
    try:
        genius = lyricsgenius.Genius(cfg['genius_token'], verbose=False,
                                     remove_section_headers=False, timeout=15)
        genius.skip_non_songs = True
        song = genius.search_song(query)
        if song and song.lyrics:
            out['song_found']   = song.title
            out['artist_found'] = song.artist
            sections = _parse_genius_sections(song.lyrics)
            artists  = sorted(set(s['artist'] for s in sections if s['artist']))
            out['attributed_singers'] = artists
            out['has_attribution']    = len(artists) >= 2
            if len(artists) < 2:
                out['warning'] = ('Lyrics found but no per-singer attribution. '
                    'Check genius.com for this song — '
                    'headers need the format: [Verse 1: Singer Name]')
        else:
            out['song_found'] = None
            out['error'] = f'Song not found for "{query}". Try a different search.'
    except Exception as e:
        out['error'] = str(e)
    return jsonify(out)


if __name__=='__main__':
    import webbrowser
    print('\n'+'='*46+'\n   StemSplit  |  http://localhost:5000\n'+'='*46+'\n')
    webbrowser.open('http://localhost:5000')
    app.run(debug=False,port=5000,host='127.0.0.1')
