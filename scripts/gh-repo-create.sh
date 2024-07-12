#!/usr/bin/env bash

set -euo pipefail

PROJECT_NAME=${PROJECT_NAME:-frochi}

main() {
	local project=""
	local project_path=""
	project_path=$(dirname "$PWD")
	project=$(basename "$project_path")
	cd "$project_path"
	git init -b main
	gh repo create "$project" \
		--private \
		--source=.
	git add .
	git commit -m 'initial commit'
}

main
