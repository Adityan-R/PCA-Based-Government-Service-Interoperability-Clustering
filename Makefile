.PHONY: run test app install clean

PYTHON = python

run:
	$(PYTHON) src/pipeline.py

test:
	$(PYTHON) -m pytest tests/ -v

app:
	streamlit run app.py

install:
	$(PYTHON) -m pip install -r requirements.txt

clean:
	powershell -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue data\processed\*, __pycache__, src\__pycache__, tests\__pycache__, .pytest_cache"
