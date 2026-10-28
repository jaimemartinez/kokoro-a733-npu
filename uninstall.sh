#!/bin/bash
# Removes everything install.sh added: /opt/kokoro-npu, the kokoro-npu command and the kernel setting.
set -eu
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO=sudo
$SUDO rm -rf /opt/kokoro-npu /usr/local/bin/kokoro-npu /etc/sysctl.d/90-kokoro-npu.conf
echo "kokoro-npu removed. (The memory setting returns to the system default after a reboot.)"
