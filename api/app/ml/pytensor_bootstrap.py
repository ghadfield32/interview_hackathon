"""
Guarantee PyTensor sees safe flags **before** first import anywhere.
Must be imported very early (e.g. from api/app/__init__.py).
"""

import os
import platform
from pathlib import Path
from .utils import find_compiler   # reuse existing helper

# ── 1️⃣  Compiler path ---------------------------------------------------------
cxx = find_compiler()
if cxx:
    print(f"🔍 Found compiler: {cxx}")
    # For MSVC, we need to quote the path properly
    if " " in cxx:
        cxx_q = f'"{cxx}"'
        print(f"📝 Quoted path: {cxx_q}")
    else:
        cxx_q = cxx
        print(f"📝 Unquoted path: {cxx_q}")
    os.environ.setdefault("PYTENSOR_CXX", cxx_q)
    print(f"✅ Set PYTENSOR_CXX to: {os.environ['PYTENSOR_CXX']}")
else:
    print("⚠️ No compiler found")

# ── 2️⃣  Safe optimiser flags --------------------------------------------------
# `fast_compile` avoids aggressive graph rewrites that inject GCC flags
os.environ.setdefault("PYTENSOR_FLAGS", "mode=FAST_COMPILE,optimizer=fast_compile")

# ── 3️⃣  Windows-specific MSVC safety -----------------------------------------
if platform.system() == "Windows" and cxx and "cl" in Path(cxx).name.lower():
    # MSVC-specific flags to prevent GCC flag injection
    os.environ.setdefault("PYTENSOR_CXXFLAGS", "/wd4100 /wd4244 /wd4267 /wd4996")
    os.environ.setdefault("THEANO_CXXFLAGS", "/wd4100 /wd4244 /wd4267 /wd4996") 