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
		if ! gcloud init --console-only; then
			return 1
		else
			return 0
		fi
	}

	gcloud-auth-adc() {
		if ! gcloud auth application-default login --no-launch-browser; then
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

	project-pyproject-toml() {
		cat <<-EOF
			[project]
			name = "$PROJECT_NAME"
			version = "0.0.0"
			requires-python = ">=3.10,<3.13"
			description = "GenJax project."
			dependencies = [
			    "genjax == $GENJAX_VERSION",
			    "genstudio == $GENSTUDIO_VERSION",
			    "jax >= 0.4.28",
			    "tensorflow-probability >= 0.23.0,<0.24",
			    "msgpack >= 1.0.8",
			    "penzai"
			]

			[tool.pixi.project]
			channels = ["conda-forge"]
			platforms = ["linux-64", "osx-arm64", "osx-64"]

			[tool.pixi.dependencies]
			jaxtyping = ">=0.2.30,<0.3"
			beartype = ">=0.18.5,<0.19"
			deprecated = ">=1.2.14,<1.3"

			[tool.pixi.pypi-dependencies]
			$PROJECT_NAME = { path = ".", editable = true }

			[tool.pixi.pypi-options]
			extra-index-urls = [
			    "https://oauth2accesstoken@us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple",
			]

			[project.optional-dependencies]
			base = [
			    "genjax == $GENJAX_VERSION",
			    "genstudio == $GENSTUDIO_VERSION",
			    "jax >= 0.4.28",
			    "tensorflow-probability >= 0.23.0,<0.24",
			    "msgpack >= 1.0.8",
			    "penzai",
			    "jaxtyping >=0.2.30,<0.3",
			    "beartype >=0.18.5,<0.19",
			    "deprecated >=1.2.14,<1.3"
			]
			cpu = ["jax[cpu] >= 0.4.28"]
			cuda = ["jax[cuda12] >= 0.4.28"]
			dev = [
			    "coverage",
			    "nbmake",
			    "pytest",
			    "pytest-benchmark",
			    "pytest-xdist[psutil]",
			    "xdoctest",
			    "matplotlib",
			    "mypy",
			    "ruff",
			    "safety",
			    "seaborn",
			    "jupyterlab",
			    "jupytext",
			    "quarto",
			]
			[tool.pixi.tasks]

			[tool.pixi.environments]
			default = { solve-group = "default" }
			cpu = { features = ["base", "cpu", "dev"] }
			gpu = { features = ["base", "cuda", "dev"] }

			[tool.pixi.feature.dev.tasks]
			demo = { cmd = [
			    "python",
			    "demo.py",
			], cwd = "src/$PROJECT_NAME", description = "Run GenJax demo" }
			notebook = { cmd = [
			    "pixi",
			    "run",
			    "jupyter",
			    "lab",
			    "--ip=0.0.0.0",
			    "--allow-root",
			    "notebooks",
			], description = "Run notebooks" }
			test = { cmd = [
			    "pytest",
			    "--benchmark-disable",
			    "--ignore",
			    "scratch",
			    "--ignore",
			    "notebooks",
			    "-n",
			    "auto",
			], description = "Run tests" }
			coverage = { cmd = [
			    "coverage",
			    "run",
			    "-m",
			    "pytest",
			    "--benchmark-disable",
			    "--ignore",
			    "scratch",
			], description = "Run coverage" }
			benchmark = { cmd = [
			    "coverage",
			    "run",
			    "-m",
			    "pytest",
			    "--benchmark-warmup",
			    "on",
			    "--ignore",
			    "tests",
			    "--benchmark-disable-gc",
			    "--benchmark-min-rounds",
			    "5000",
			], description =  "Run benchmarks" }
			docs-build = { cmd = [
			    "quarto",
			    "render",
			    "notebooks",
			    "--execute",
			] }
			docs-serve = { cmd = [
			    "python",
			    "-m",
			    "http.server",
			    "8080",
			    "--bind",
			    "127.0.0.1",
			    "--directory",
			    "site",
			] }
			docs = { depends-on = ["docs-build", "docs-serve"], description = "Build and serve the docs" }

			[tool.pixi.target.osx-arm64]
			build-dependencies = { scipy = { version = "1.14.0.*" }, numpy = { version = "1.26.4.*" } }

			[tool.pixi.feature.cpu]
			platforms = ["linux-64", "osx-64", "osx-arm64"]

			[tool.pixi.feature.cuda]
			platforms = ["linux-64"]
			system-requirements = { cuda = "12.4" }

			[tool.pixi.feature.cuda.target.linux-64]
			pypi-dependencies = { jax = { version = ">=0.4.28", extras = ["cuda12"] } }

			[tool.coverage.paths]
			source = ["src", "*/site-packages"]
			tests = ["tests", "*/tests"]

			[tool.coverage.run]
			omit = [".*", "*/site-packages/*"]

			[tool.coverage.report]
			show_missing = true
			fail_under = 45

			[tool.pyright]
			venvPath = "."
			venv = ".pixi"
			include = ["src", "tests"]
			exclude = ["**/__pycache__"]
			defineConstant = { DEBUG = true }
			reportMissingImports = true
			reportMissingTypeStubs = false

			[tool.ruff]
			exclude = [
			    ".bzr",
			    ".direnv",
			    ".eggs",
			    ".git",
			    ".git-rewrite",
			    ".hg",
			    ".mypy_cache",
			    ".nox",
			    ".pants.d",
			    ".pytype",
			    ".ruff_cache",
			    ".svn",
			    ".tox",
			    ".venv",
			    "__pypackages__",
			    "_build",
			    "buck-out",
			    "build",
			    "dist",
			    "node_modules",
			    "venv",
			]
			extend-include = ["*.ipynb"]
			line-length = 88
			indent-width = 4

			[tool.ruff.lint.pydocstyle]
			convention = "google"

			[tool.ruff.lint]
			preview = true
			extend-select = ["I"]
			select = ["E4", "E7", "E9", "F"]
			# F403 disables errors from $() imports, which we currently use heavily.
			ignore = ["F403"]
			fixable = ["ALL"]
			unfixable = []
			dummy-variable-rgx = "^(_+|(_+[a-zA-Z0-9_]*[a-zA-Z0-9]+?))$"

			[tool.ruff.format]
			preview = true
			skip-magic-trailing-comma = false
			docstring-code-format = true
			quote-style = "double"
			indent-style = "space"
			line-ending = "auto"

			[tool.mypy]
			strict = true
			warn_unreachable = true
			pretty = true
			show_column_numbers = true
			show_error_codes = true
			show_error_context = true
		EOF
	}

	project-src-init() {
		cat <<-EOF
			def hello():
			    return "world!"
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

			        return jnp.mean(p_chm["p"])

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

	project-init() {
		gh repo create "$PROJECT_NAME" \
			--private \
			--clone \
			--add-readme \
			--gitignore Python

		pushd "$PROJECT_NAME"

		if ! mkdir -p "src/$PROJECT_NAME"; then
			exit 1
		fi
		if ! mkdir -p "tests"; then
			exit 1
		fi
		if ! mkdir -p ".pixi"; then
			exit 1
		fi

		echo "pixi.lock linguist-language=YAML linguist-generated=true" >>.gitattributes
		echo ".pixi/envs" >>.gitignore
		project-pyproject-toml >pyproject.toml

		pushd .pixi
		project-pixi-config >config.toml
		popd

		pushd "src/$PROJECT_NAME"
		project-src-init >__init__.py
		project-src-demo >demo.py
		popd

		pushd tests
		project-tests-test >test.py
		popd

		popd
	}

	dev-environment-init() {
		echo "HOME $HOME"

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

		# install gh
		echo "installing gh..."
		if ! gh-global-install; then
			echo "gh install failed"
			exit 1
		fi
		v=$(gh --version)
		p=$(which gh)
		printf "  ✓ gh %s installed (%s)\n\n" "$v" "$p"

		# install gcloud
		echo "installing gcloud..."
		if ! gcloud-global-install; then
			echo "gcloud install failed"
			exit 1
		fi
		v=$(gcloud --version)
		p=$(which gcloud)
		printf "  ✓ gcloud %s installed (%s)\n\n" "$v" "$p"

		# initialize and authenticate gcloud
		echo "init gcloud..."
		if ! gcloud-init; then
			echo "gcloud not initialized"
			exit 1
		fi
		echo "authenticating gcloud credentials..."
		if ! gcloud-auth-adc; then
			echo "gcloud not authenticated"
			exit 1
		fi
		printf "  ✓ gcloud initialized and authenticated\n\n"

		# initialize project
		echo "initializing project..."
		if ! project-init; then
			echo "couldn't initialize project"
			exit 1
		fi
		printf "  ✓ project initialized\n\n"

		# pixi install
		echo "installing project environments..."
		pushd "$PROJECT_NAME"
		if ! pixi install; then
			echo "couldn't initialize project"
			exit 1
		fi
		popd
		printf "  ✓ project environments installed\n\n"

		printf "\nbootstrap complete! run this command and you're done:\n"
		printf "  → source %s\n" "$shell_config\n"
	}

	dev-environment-init
}
__wrap__
