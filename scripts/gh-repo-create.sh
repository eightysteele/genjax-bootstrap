#!/usr/bin/env bash

set -euo pipefail

PROJECT_NAME=${PROJECT_NAME:-frochi}

main() {
	local project=""
	project=$(dirname "$PWD")
	git init -b main
	gh repo create "$project" \
		--private \
		--source=.
	git add .
	git commit -m 'initial commit'
}

main
