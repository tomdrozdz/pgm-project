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
	@echo -e "    env - create Python environment"
	@echo -e "    crf - run experiments for CRF"
	@echo -e "    bnn - run experiments for Bayesian Neural Network"
	@echo -e "    lstm - run experiments for Bayesian LSTM"

${_ENV_CREATED}:
	@rm -rf "${VENV_DIR}"
	@${PYTHON} -m venv "${VENV_DIR}"
	@touch $@

${_ENV_READY}: ${_ENV_CREATED}
	@./${VENV_DIR}/bin/pip install --upgrade -r requirements.txt
	@./${VENV_DIR}/bin/pip install torch torchvision torchaudio \
		--extra-index-url https://download.pytorch.org/whl/cu113

	@touch $@

env: ${_ENV_READY}

crf: env
	@./"${VENV_DIR}"/bin/python crf_run.py

bnn: env
	@./"${VENV_DIR}"/bin/python bnn_run.py

lstm: env
	@./"${VENV_DIR}"/bin/python lstm_hparams.py
	@./"${VENV_DIR}"/bin/python lstm_params_charts.py
	@./"${VENV_DIR}"/bin/python lstm_evaluation.py
