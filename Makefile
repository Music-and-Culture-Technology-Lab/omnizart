CHECK_SRC := ./omnizart


.PHONY: all
all: lint test

# --------------------------------------------------------------
# Linter
# --------------------------------------------------------------

.PHONY: lint
lint: check-flake check-pylint

.PHONY: check-flake
check-flake:
	@echo "Checking with flake..."
	@flake8 --config .config/flake $(CHECK_SRC)

.PHONY: check-pylint
check-pylint:
	@echo "Checking with pylint..."
	@pylint --fail-under 9.5 --rcfile .config/pylintrc $(CHECK_SRC)

.PHONY: check-black
check-black:
	@echo "Checking with black..."
	@black --check $(CHECK_SRC)

.PHONY: format
format:
	@echo "Format code with black"
	@black $(CHECK_SRC)
	@echo "Format code with yapf"
	@yapf $(CHECK_SRC) --in-place --recursive --style .config/yapf.style

# --------------------------------------------------------------
# Unittest
# --------------------------------------------------------------

.PHONY: test
test:
	@echo "Run unit tests"
	@pytest --cov-fail-under=25 --cov-report=html --cov=omnizart tests

# --------------------------------------------------------------
# Other convenient utilities
# --------------------------------------------------------------

.PHONY: export
export:
	@echo "Exporting requirements.txt and setup.py"
	@poetry export -f requirements.txt -o requirements.txt
	@./scripts/create_setup.sh

.PHONY: install
install:
	@./scripts/install.sh venv

.PHONY: install-dev
install-dev:
	@poetry config virtualenvs.create false
	@poetry install
	@./scripts/gdrive.sh https://drive.google.com/file/d/10i8z1zH60a2coKEst47lELdkvZUmgd1b/view?usp=sharing
	@unzip resource.zip
	@mv resource/* tests/resource
	@rm -d resource
	@rm resource.zip

.PHONY: clean
clean:
	@rm -rf .venv/
	@rm -rf ~/.cache/pypoetry/


.PHONY: build-docker
build-docker:
	docker build --network=host -t omnizart:dev ./
