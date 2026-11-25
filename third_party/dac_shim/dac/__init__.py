"""Minimal shim of descript-audio-codec (MIT, (c) Descript).

Seed-VC only imports ``dac.nn.quantize``. The real package pulls in
``descript-audiotools`` which pins protobuf==3.19.6 and breaks modern
environments, so only the two MIT-licensed source files are vendored here.
"""
__version__ = "1.0.0-shim"
