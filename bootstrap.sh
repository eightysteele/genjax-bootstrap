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

	project-notebook-demo() {
		cat <<-EOF
			{
			 "cells": [
			  {
			   "cell_type": "code",
			   "execution_count": null,
			   "id": "ebbdee30-9a85-4ea9-abb2-be64e7a83bb1",
			   "metadata": {},
			   "outputs": [],
			   "source": [
			    "import timeit\n",
			    "\n",
			    "import jax\n",
			    "import jax.numpy as jnp\n",
			    "import genjax\n",
			    "from genjax import beta, flip, gen, Target, ChoiceMap\n",
			    "from genjax.inference.smc import ImportanceK\n",
			    "\n",
			    "@gen\n",
			    "def beta_bernoulli(α, β):\n",
			    "    \"\"\"Define the generative model.\"\"\"\n",
			    "    p = beta(α, β) @ 'p'\n",
			    "    v = flip(p) @ \"v\"\n",
			    "    return v\n",
			    "\n",
			    "def run_inference(obs: bool, platform='cpu'):\n",
			    "    \"\"\"Estimate $(p) over 50 independent trials of SIR (K = 50 particles).\"\"\"\n",
			    "    # Set the device\n",
			    "    device = jax.devices(platform)[0]\n",
			    "    key = jax.random.PRNGKey(314159)\n",
			    "    key = jax.device_put(key, device)\n",
			    "\n",
			    "    # JIT compilation will be target-specific based on device (CPU or GPU)\n",
			    "    @jax.jit\n",
			    "    def execute_inference():\n",
			    "        # Inference query with the a model, arguments, and constraints\n",
			    "        posterior = Target(beta_bernoulli, (2.0, 2.0), ChoiceMap.d({\"v\": obs}))\n",
			    "\n",
			    "        # Use a library algorithm, or design your own—more on that in the docs!\n",
			    "        alg = ImportanceK(posterior, k_particles=50)\n",
			    "\n",
			    "        # Everything is JAX compatible by default—jit, vmap, etc.\n",
			    "        skeys = jax.random.split(key, 50)\n",
			    "        _, p_chm = jax.vmap(\n",
			    "            alg.random_weighted, in_axes=(0, None))(skeys, posterior)\n",
			    "\n",
			    "        return jnp.mean(p_chm['p'])\n",
			    "\n",
			    "    return execute_inference\n",
			    "\n",
			    "\n",
			    "n = 1000\n",
			    "print(f\"\Starting GenJax demo with {n} runs...\")\n",
			    "\n",
			    "# CPU compile, execute, benchmark, profile\n",
			    "cpu_jit = jax.jit(run_inference(True, 'cpu'))\n",
			    "cpu_jit().block_until_ready()\n",
			    "ms = timeit.timeit(\n",
			    "    'cpu_jit()',\n",
			    "    globals=globals(),\n",
			    "    number=n\n",
			    ") / n * 1000\n",
			    "print(f\"CPU: Average runtime over {n} runs = {ms} (ms)\")\n",
			    "\n",
			    "# GPU compile, execute, benchmark, profile\n",
			    "gpu_jit = jax.jit(run_inference(True, 'cpu'))\n",
			    "gpu_jit().block_until_ready()\n",
			    "try:\n",
			    "    jax.devices('gpu')\n",
			    "    ms = timeit.timeit(\n",
			    "        'gpu_jit',\n",
			    "        globals=globals(),\n",
			    "        number=n\n",
			    "    ) / n * 1000\n",
			    "    print(f\"GPU: Average runtime over {n} runs = {ms} (ms)\")\n",
			    "except RuntimeError as e:\n",
			    "    print(\"(No GPU device on host)\")\n",
			    "\n",
			    "print(\"Done!\")"
			   ]
			  }
			 ],
			 "metadata": {
			  "kernelspec": {
			   "display_name": "Python 3 (ipykernel)",
			   "language": "python",
			   "name": "python3"
			  },
			  "language_info": {
			   "codemirror_mode": {
			    "name": "ipython",
			    "version": 3
			   },
			   "file_extension": ".py",
			   "mimetype": "text/x-python",
			   "name": "python",
			   "nbconvert_exporter": "python",
			   "pygments_lexer": "ipython3",
			   "version": "3.12.4"
			  }
			 },
			 "nbformat": 4,
			 "nbformat_minor": 5
			}
		EOF
	}

	project-src-demo() {
		cat <<-EOF
			import timeit

			import jax
			import jax.numpy as jnp
			import genjax
			from genjax import beta, flip, gen, Target, ChoiceMap
			from genjax.inference.smc import ImportanceK

			@gen
			def beta_bernoulli(α, β):
			    """Define the generative model."""
			    p = beta(α, β) @ "p"
			    v = flip(p) @ "v"
			    return v

			def run_inference(obs: bool, platform='cpu'):
			    """Estimate $(p) over 50 independent trials of SIR (K = 50 particles)."""
			    # Set the device
			    device = jax.devices(platform)[0]
			    key = jax.random.PRNGKey(314159)
			    key = jax.device_put(key, device)

			    # JIT compilation will be target-specific based on device (CPU or GPU)
			    @jax.jit
			    def execute_inference():
			        # Inference query with the a model, arguments, and constraints
			        posterior = Target(beta_bernoulli, (2.0, 2.0), ChoiceMap.d({"v": obs}))

			        # Use a library algorithm, or design your own—more on that in the docs!
			        alg = ImportanceK(posterior, k_particles=50)

			        # Everything is JAX compatible by default—jit, vmap, etc.
			        skeys = jax.random.split(key, 50)
			        _, p_chm = jax.vmap(
			            alg.random_weighted, in_axes=(0, None))(skeys, posterior)

			        return jnp.mean(p_chm['p'])

			    return execute_inference


			n = 1000
			print(f"\nStarting GenJax demo with {n} runs...")

			# CPU compile, execute, benchmark, profile
			cpu_jit = jax.jit(run_inference(True, 'cpu'))
			cpu_jit().block_until_ready()
			ms = timeit.timeit(
			    'cpu_jit()',
			    globals=globals(),
			    number=n
			) / n * 1000
			print(f"CPU: Average runtime over {n} runs = {ms} (ms)")

			# GPU compile, execute, benchmark, profile
			gpu_jit = jax.jit(run_inference(True, 'cpu'))
			gpu_jit().block_until_ready()
			try:
			    jax.devices('gpu')
			    ms = timeit.timeit(
			        'gpu_jit',
			        globals=globals(),
			        number=n
			    ) / n * 1000
			    print(f"GPU: Average runtime over {n} runs = {ms} (ms)")
			except RuntimeError as e:
			    print("(No GPU device on host)")

			print("Done!")

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

	pixi-hardcode() {
		cat <<-EOF
			[tool.pixi.target.osx-arm64]
			build-dependencies = { scipy = { version = "1.14.0.*" }, numpy = { version = "1.26.4.*" } }

			[tool.pixi.environments]
			default = { solve-group = "default" }
			cpu = ["cpu", "dev"]
			cuda = ["cuda", "dev"]

			[tool.pixi.feature.cpu]
			platforms = ["linux-64", "osx-64", "osx-arm64"]

			[tool.pixi.feature.cpu.pypi-dependencies]
			jax = { version = ">=0.4.28", extras = ["cpu"] }
			genjax = "*"
			genstudio = "*"
			tensorflow-probability = ">=0.23.0"
			penzai = ">=0.1.1"
			msgpack = ">=1.0.8"

			[tool.pixi.feature.cuda]
			platforms = ["linux-64", "osx-arm64"]
			system-requirements = { cuda = "12.4" }

			[tool.pixi.feature.cuda.target.linux-64.pypi-dependencies]
			jax = { version = ">=0.4.28", extras = ["cuda12"] }
			genjax = "*"
			genstudio = "*"
			tensorflow-probability = ">=0.23.0"
			penzai = ">=0.1.1"
			msgpack = ">=1.0.8"
		EOF
	}

	pixi-tools-hardcode() {
		cat <<-EOF
			[tool.pyright]
			venvPath = "."
			venv = ".pixi"
			pythonVersion = "3.12.3"
			include = ["src", "notebooks"]

			[tool.ruff.format]
			exclude = ["notebooks/demo.py"]

			[tool.ruff.lint.per-file-ignores]
			"notebooks/intro.py" = ["E999"]

			[tool.jupytext]
			formats = "ipynb,py:percent"
		EOF
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
			"jupyter lab --ip=0.0.0.0 --allow-root notebooks" \
			--feature dev \
			--description "run notebooks"

		pixi task add test \
			"pytest --benchmark-disable --ignore scratch --ignore notebooks/demo.ipynb -n auto" \
			--feature dev \
			--description "run tests"

		pixi-tools-hardcode >>pyproject.toml

		pixi install

		popd &>/dev/null
	}

	prompt-user() {
		local os=$(uname -s)
		local arch=$(uname -m)

		printf "This script will bootstrap a new project into a virtual GenJax development environment for %s %s.\n\n" $os $arch

		read -p "Do you want to continue? (y/n): " choice

		case "$choice" in
		y | Y)
			echo "Bootstrapping..."
			return 0
			;;
		n | N)
			echo "Aborting."
			return 1
			;;
		*)
			echo "Invalid input. Please enter 'y' to continue or 'n' to abort."
			exit 1
			;;
		esac
	}

	init-dev-environment() {
		if ! prompt-user; then
			exit 1
		fi

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
