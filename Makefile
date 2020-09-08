CHECK_SRC := ./omnizart


.PHONY: all
all: check test

# --------------------------------------------------------------
# Linter
# --------------------------------------------------------------

.PHONY: check
check: check-flake check-pylint

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
	@python -m pytest ./tests

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
	@./scripts/install.sh

.PHONY: clean
clean:
	@rm -rf .venv/
	@rm -rf ~/.cache/pypoetry/

