#!/usr/bin/env bash

set -euo pipefail

__wrap__() {

	PROJECT_NAME=${PROJECT_NAME:-frochi}
	GENJAX_VERSION=${GENJAX_VERSION:-0.5.0}
	GENSTUDIO_VERSION=${GENSTUDIO_VERSION:-2024.6.20.1130}

	PIXI_BIN="$HOME/.pixi/bin"
	PIPX_BIN="$HOME/.local/bin"
	export PATH=$PIXI_BIN:$PIPX_BIN:$PATH

	echo "Project: $PROJECT_NAME"
	pixi-installed() {
		if command -v pixi &>/dev/null; then
			return 0
		else
			return 1
		fi
	}

	pixi-update() {
		if pixi self-update; then
			return 0
		else
			return 1
		fi
	}

	pixi-global-install() {
		if curl -fsSL https://pixi.sh/install.sh | bash; then
			return 0
		else
			return 1
		fi
	}

	pixi-enable-autocomplete() {
		local shell=$1
		local config=$2

		case $shell in
		bash)
			echo 'eval "$(pixi completion --shell bash)"' >>"$config"
			;;
		zsh)
			echo 'eval "$(pixi completion --shell zsh)"' >>"$config"
			;;
		fish)
			echo 'pixi completion --shell fish | source' >>"$config"
			;;
		elvish)
			echo 'eval (pixi completion --shell elvish | slurp)' >>"$config"
			;;
		*)
			echo "unknown shell: $shell"
			i
			;;
		esac
	}

	pipx-installed() {
		if command -v pipx &>/dev/null; then
			return 0
		else
			return 1
		fi
	}

	pipx-global-install() {
		if pixi global install pipx; then
			return 0
		else
			return 1
		fi
	}

	pipx-install-keyring() {
		if pipx install --force keyring; then
			return 0
		else
			return 1
		fi
	}

	pipx-inject-google-artifact-registry() {
		if pipx inject --force keyring \
			keyrings.google-artifactregistry-auth \
			--index-url https://pypi.org/simple \
			--force; then
			return 0
		else
			return 1
		fi
	}

	git-installed() {
		if command -v git &>/dev/null; then
			return 0
		else
			return 1
		fi
	}

	git-global-install() {
		if pixi global install git; then
			return 0
		else
			return 1
		fi
	}

	gh-installed() {
		if command -v gh &>/dev/null; then
			return 0
		else
			return 1
		fi
	}

	gh-global-install() {
		if pixi global install gh; then
			return 0
		else
			return 1
		fi
	}

	gh-login() {
		if gh auth login; then
			return 0
		else
			return 1
		fi
	}

	gcloud-global-install() {
		if pixi global install google-cloud-sdk; then
			return 0
		else
			return 1
		fi
	}

	gcloud-authenticated() {
		if gcloud auth print-access-token &>/dev/null; then
			return 0
		else
			return 1
		fi
	}

	gcloud-init() {
		if ! gcloud init; then
			return 1
		else
			return 0
		fi
	}

	gcloud-auth-adc() {
		if ! gcloud auth application-default login; then
			return 1
		fi
		return 0
	}

	get-current-shell() {
		local path=""
		local name=""

		if [ -n "$SHELL" ]; then
			path=$(echo $SHELL)
			name=$(basename $path)
			echo $name
			return 0
		else
			return 1
		fi
	}

	get-shell-config() {
		local shell=$1
		case $shell in
		bash)
			echo "$HOME/.bashrc"
			;;
		zsh)
			echo "$HOME/.zshrc"
			;;
		fish)
			echo "$HOME/.config/fish/config.fish"
			;;
		elvish)
			echo "$HOME/.elvish/rc.elv"
			;;
		*)
			echo "unknown shell: $shell"
			;;
		esac
	}

	init-project-repo() {
		git init -b main
		gh repo create "$PROJECT_NAME" \
			--private \
			--source=.
		git add .
		git commit -m 'initial commit'
	}

	rename-project() {
		if [[ "$OSTYPE" == "darwin"* ]]; then
			sed -i '' "s/frochi/$PROJECT_NAME/g" pyproject.toml
			sed -i '' "s/frochi/$PROJECT_NAME/g" tests/test_frochi.py
		else
			sed -i "s/frochi/$PROJECT_NAME/g" pyproject.toml
			sed -i "s/frochi/$PROJECT_NAME/g" tests/test_frochi.py
		fi
		mv src/frochi "src/$PROJECT_NAME"
		mv tests/test_frochi.py "tests/test_$PROJECT_NAME.py"
	}

	init-dev-environment() {
		local shell=""
		local shell_config=""
		local v=""
		local p=""

		# check shell
		echo "checking shell..."
		if ! shell=$(get-current-shell); then
			echo "SHELL not set"
			exit 1
		else
			printf "  ✓ shell is %s \n" "$shell"
		fi
		shell_config=$(get-shell-config $shell)
		if [ -e "$shell_config" ]; then
			printf "  ✓ shell config: %s \n\n" "$shell_config"
		else
			echo "SHELL config not found"
			exit 1
		fi

		# install pixi
		echo "checking pixi..."
		if ! pixi-installed; then
			echo "  installing pixi..."
			if ! pixi-global-install; then
				echo "couldn't install pixi"
				exit 1
			else
				pixi-enable-autocomplete "$shell" "$shell_config"
			fi
		else
			pixi self-update
		fi
		v=$(pixi --version)
		p=$(which pixi)
		printf "  ✓ %s installed (%s)\n\n" "$v" "$p"

		# install git
		echo "installing git..."
		if ! git-global-install; then
			echo "git install failed"
			exit 1
		fi
		v=$(git --version)
		p=$(which git)
		printf "  ✓ git %s installed (%s)\n\n" "$v" "$p"

		# install gh
		echo "installing gh..."
		if ! gh-global-install; then
			echo "gh install failed"
			exit 1
		fi
		v=$(gh --version)
		p=$(which gh)
		printf "  ✓ gh %s installed (%s)\n\n" "$v" "$p"

		gh repo clone \
			eightysteele/genjax-bootstrap \
			"$PROJECT_NAME"

		pushd "$PROJECT_NAME" &>/dev/null
		#rm -rf .git
		#rename-project

		#echo "running 'pixi install --frozen'..."
		#pixi install --frozen

		pushd scripts &>/dev/null
		./check-auth.sh
		popd &>/dev/null

		echo "running 'pixi task list'..."
		pixi task list

		printf "\nbootstrap complete! run these commands:\n"
		printf "  1) source %s\n" "$shell_config"
		printf "  2) cd %s\n" "$PROJECT_NAME"
		printf "  3) pixi run demo\n\n"
	}

	init-dev-environment
}
__wrap__
