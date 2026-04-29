#!/bin/bash
set -euo pipefail

REPO_URL="${REPO_URL:?REPO_URL must be set}"
REPO_DIR="${REPO_DIR:?REPO_DIR must be set}"
COMMAND="${COMMAND:?COMMAND must be set}"

if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
else
  git -C "${REPO_DIR}" pull --ff-only
fi

cd "${REPO_DIR}"
exec bash -lc "${COMMAND}"
