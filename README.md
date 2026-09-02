Remember to download the bitstream, driver and library to the FPGA!

It may be necessary to append ` -- -l $(($(nproc) / 2))` to the `cmake --build`
commands to avoid the WSL VM using too much memory `VmmemWSL` and `hv_balloon`
start trashing the drive that host the VM hard disk and this slows things down
to a crawl.

```
# To build IREE
sudo apt install cmake ninja-build clang lld libstdc++-13-dev-armhf-cross

# To build tosa-converter-for-tflite

sudo apt install python3-venv

# from https://bazel.build/install/ubuntu#add-dis-uri
sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://releases.bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list

sudo apt update && sudo apt install bazel-7.4.1
```
