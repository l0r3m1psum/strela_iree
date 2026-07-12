
[ ! -d venv ] && python3 -m venv venv

. venv/bin/activate

PATH="$PWD/iree-build/tools:$PATH"
