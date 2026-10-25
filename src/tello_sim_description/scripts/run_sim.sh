#!/usr/bin/env bash



set -e
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GZ_SIM_RESOURCE_PATH="$PKG_DIR/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
WORLD="$PKG_DIR/worlds/tello_land.sdf"

if [[ "$1" == "--headless" ]]; then
  exec gz sim -s -r "$WORLD"
else
  exec gz sim -r --gui-config "$PKG_DIR/config/gui.config" "$WORLD"
fi
