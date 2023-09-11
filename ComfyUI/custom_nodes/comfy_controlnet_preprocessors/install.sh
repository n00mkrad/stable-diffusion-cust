#!/bin/bash
if [ -e "../../../python_embeded/python" ]; then
    echo Custom Python build of ComfyUI standalone executable detected:
    echo "$(readlink -f "../../python_embeded/python")"
    echo --------------------------------------------------
    ../../../python_embeded/python install.py ${1:+"$1"} #One-liner is za bezt
else
    echo "Custom Python not found. Use system's Python executable instead:"
    echo "$(which python3)"
    echo --------------------------------------------------
    python3 install.py ${1:+"$1"}
fi
