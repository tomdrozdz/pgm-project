.PHONY: env \
		crf \
		bnn \
		lstm

PYTHON ?= python
VENV_DIR := .venv
_ENV_CREATED = ${VENV_DIR}/.created
_ENV_READY = ${VENV_DIR}/.ready
.DEFAULT_GOAL = help

help:
	@echo -e "Available targets:"
	@echo -e \t env - create Python environment
	@echo -e \t crf - run experiments for CRF
	@echo -e \t bnn - run experiments for Bayesian Neural Network
	@echo -e \t lstm - run experiments for Bayesian LSTM

${_ENV_CREATED}:
	@rm -rf "${VENV_DIR}"
	@${PYTHON} -m venv "${VENV_DIR}"
	@touch $@

${_ENV_READY}: ${_ENV_CREATED}
	@./${VENV_DIR}/bin/pip install --upgrade -r requirements.txt
	@touch $@

env: ${_ENV_READY}

crf: env
	@./"${VENV_DIR}"/bin/python run_crf.py

bnn: env
	@./"${VENV_DIR}"/bin/python run_

lstm: env
	@./"${VENV_DIR}"/bin/python run_
