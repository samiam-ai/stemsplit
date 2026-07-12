@echo off
title StemSplit - Setup
color 0A
echo.
echo  ================================================
echo    StemSplit - First-Time Setup
echo  ================================================
echo.
echo  Installing Flask...
pip install flask
echo.
echo  Installing Demucs (stem separation AI)...
pip install demucs
echo.
echo  Installing TorchCodec (audio backend)...
pip install torchcodec
echo.
echo  Installing pydub (stem mixer)...
pip install pydub
echo.
echo  Installing audioop-lts (Python 3.13 compatibility)...
pip install audioop-lts
echo.
echo  Installing pedalboard (Spotify audio effects)...
pip install pedalboard
echo.
echo  Installing noisereduce (artifact cleanup)...
pip install noisereduce
echo.
echo  Installing pyloudnorm (mastering normalization)...
pip install pyloudnorm
echo.
echo  Installing yt-dlp (YouTube to MP3 converter)...
pip install yt-dlp
echo.
echo  Installing resemblyzer (voice fingerprinting for auto-classify)...
pip install resemblyzer
echo.
echo  Installing audio-separator (lead vs backing vocal split)...
pip install "audio-separator[cpu]" onnxruntime
echo.
echo  ================================================
echo    Setup complete! Run start.bat to launch.
echo  ================================================
echo.
pause
