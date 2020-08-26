CHECK_SRC := ./omnizart


.PHONY: all
all: check test

# --------------------------------------------------------------
# Linter
# --------------------------------------------------------------

.PHONY: check
check: check-flake check-pylint check-black

.PHONY: check-flake
check-flake:
	@echo "Checking with flake..."
	@flake8 --config .config/flake $(CHECK_SRC)

.PHONY: check-pylint
check-pylint:
	@echo "Checking with pylint..."
	@pylint --rcfile .config/pylintrc $(CHECK_SRC)

.PHONY: check-black
check-black:
	@echo "Checking with black..."
	@black --check $(CHECK_SRC)

.PHONY: format
format:
	@echo "Format code with black"
	@black $(CHECK_SRC)

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
	@echo "Exporting requirements.txt"
	@poetry export -f requirements.txt -o requirements.txt

.PHONY: install
install:
	@./scripts/install.sh

.PHONY: clean
clean:
	@rm -rf .venv

