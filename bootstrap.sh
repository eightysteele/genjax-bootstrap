#!/usr/bin/env bash

set -euo pipefail

__wrap__() {

	PROJECT_NAME=${PROJECT_NAME:-frochi}
	GENJAX_VERSION=${GENJAX_VERSION:-0.5.0}
	GENSTUDIO_VERSION=${GENSTUDIO_VERSION:-2024.6.20.1130}

	PIXI_BIN="$HOME/.pixi/bin"
	PIPX_BIN="$HOME/.local/bin"
	export PATH=$PIXI_BIN:$PIPX_BIN:$PATH

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
		if gcloud auth print-access-token >/dev/null 2>&1; then
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

	project-src-init() {
		cat <<-EOF
			def hello():
			    return "world!"
		EOF
	}

	project-tests-test() {
		cat <<-EOF
			from $PROJECT_NAME import hello

			def test_$PROJECT_NAME():
			    assert hello() == ("world!")
		EOF
	}

	project-pixi-config() {
		cat <<-EOF
			[pypi-config]
			index-url = "https://pypi.org/simple"
			extra-index-urls = ["https://oauth2accesstoken@us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple"]
			keyring-provider = "subprocess"
		EOF
	}

	init-project-repo() {
		git init -b main
		gh repo create "$PROJECT_NAME" \
			--private \
			--source=.
		git add .
		git commit -m 'initial commit'
	}

	init-project-files() {
		if ! mkdir -p "src/$PROJECT_NAME"; then
			exit 1
		fi
		if ! mkdir -p "notebooks"; then
			exit 1
		fi
		if ! mkdir -p "tests"; then
			exit 1
		fi
		if ! mkdir -p ".pixi"; then
			exit 1
		fi

		echo ".pixi/envs" >>.gitignore
		project-pixi-config >>.pixi/config.toml

		curl -fsSL \
			https://raw.githubusercontent.com/eightysteele/genjax-bootstrap/main/pyproject.toml \
			>pyproject.toml

		curl -fsSL \
			https://raw.githubusercontent.com/eightysteele/genjax-bootstrap/main/pixi.lock \
			>pixi.lock

		pushd "src/$PROJECT_NAME" &>/dev/null
		project-src-init >__init__.py
		curl -fsSL \
			https://raw.githubusercontent.com/eightysteele/genjax-bootstrap/main/demo.py \
			>demo.py
		popd &>/dev/null

		pushd tests &>/dev/null
		project-tests-test >test.py
		popd &>/dev/null

		pushd notebooks &>/dev/null
		curl -fsSL \
			https://raw.githubusercontent.com/eightysteele/genjax-bootstrap/main/demo.ipynb \
			>demo.ipynb
		popd &>/dev/null
	}

	init-project() {
		# config
		pixi config \
			set \
			--global \
			pypi-config.index-url \
			"https://pypi.org/simple"

		url="https://oauth2accesstoken@us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple"
		config=$(pixi config list pypi-config 2>&1)
		if ! echo "$config" | grep -q "$url"; then
			pixi config \
				append \
				--global \
				pypi-config.extra-index-urls \
				"$url"
		fi

		pixi config \
			set \
			--global \
			pypi-config.keyring-provider \
			"subprocess"

		pixi init \
			--pyproject \
			--platform linux-64 \
			--platform osx-arm64 \
			--platform osx-64 \
			--platform win-64 \
			"$PROJECT_NAME"

		pushd "$PROJECT_NAME" &>/dev/null

		# conda
		pixi add \
			"jaxtyping >=0.2.28" \
			"beartype >=0.18.5" \
			"deprecated >=1.2.14"

		# dev feature
		echo 'add dev'
		pixi add \
			--feature dev \
			"nbmake >=1.4.6" \
			"pytest >=7.2.0" \
			"ruff >=0.1.3" \
			"jupyterlab >=4.1.6" \
			"jupytext >=1.16.2"

		pixi-hardcode >>pyproject.toml

		pixi install

		# environments
		# pixi project environment add \
		# 	--solve-group default \
		# 	default

		# pixi project environment add \
		# 	--feature cpu \
		# 	--feature dev \
		# 	cpu

		# pixi project environment add \
		# 	--feature cuda \
		# 	--feature dev \
		# 	gpu

		# tasks
		pixi task add demo \
			python demo.py \
			--feature dev \
			--cwd "src/$PROJECT_NAME" \
			--description "run genjax demo"

		pixi task add notebook \
			"jupyter lab --ip=0.0.0.0 --allow-root notebooks/demo.ipynb" \
			--feature dev \
			--description "run notebooks"

		pixi task add test \
			"pytest --benchmark-disable --ignore scratch --ignore notebooks -n auto" \
			--feature dev \
			--description "run tests"

		pixi-tools-hardcode >>pyproject.toml

		pixi install

		popd &>/dev/null
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

		# install pipx
		echo "  installing pipx..."
		if ! pipx-global-install; then
			echo "couldn't install pipx"
			exit 1
		fi
		pipx ensurepath --force
		v=$(pipx --version)
		p=$(which pipx)
		printf "  ✓ pipx %s installed (%s)\n\n" "$v" "$p"

		# install keyring
		echo "installing keyring..."
		if ! pipx-install-keyring; then
			echo "pipx couldn't install keyring"
			exit 1
		fi
		p=$(which keyring)
		printf "  ✓ keyring installed (%s)\n\n" "$p"

		# inject gcloud auth backend
		echo "injecting google-artifact-registry-auth backend..."
		if ! pipx-inject-google-artifact-registry; then
			echo "pipx couldn't inject google artifact registry keyring"
			exit 1
		fi
		printf "  ✓ google artifact registry backend injpfected\n\n"

		keyring --list-backends

		printf "\n\n"

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

		# # install gcloud
		# echo "installing gcloud..."
		# if ! gcloud-global-install; then
		# 	echo "gcloud install failed"
		# 	exit 1
		# fi
		# v=$(gcloud --version)
		# p=$(which gcloud)
		# printf "  ✓ gcloud %s installed (%s)\n\n" "$v" "$p"

		# # initialize and authenticate gcloud
		# #echo "init gcloud..."
		# #if ! gcloud-init; then
		# #	echo "gcloud not initialized"
		# #	exit 1
		# #fi
		# echo "authenticating gcloud credentials..."
		# if ! gcloud-auth-adc; then
		# 	echo "gcloud not authenticated"
		# 	exit 1
		# fi
		# printf "  ✓ gcloud initialized and authenticated\n\n"

		# initialize project
		echo "initializing project..."
		if ! init-project; then
			echo "couldn't initialize project"
			exit 1
		fi
		printf "  ✓ project initialized\n\n"

		pushd "$PROJECT_NAME" &>/dev/null

		# initialize project files
		echo "initializing project files..."
		if ! init-project-files; then
			echo "couldn't initialize project files"
			exit 1
		fi
		printf "  ✓ project files initialized\n\n"

		# initialize project repo
		# echo "initializing project repo..."
		# if ! init-project-repo; then
		# 	echo "couldn't initialize project repo"
		# 	exit 1
		# fi
		printf "  ✓ project repo initialized\n\n"

		pixi task list

		printf "\nbootstrap complete! run these commands:\n"
		printf "  → source %s\n" "$shell_config"
		printf "  → cd %s\n" "$PROJECT_NAME"
		printf "  → pixi run notebook\n\n"
	}

	init-dev-environment
}
__wrap__
