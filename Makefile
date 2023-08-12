run:
	PYTHONPATH=. poetry run streamlit run prompterator/main.py

lint:
	poetry run black --check .
	poetry run isort --check .

format:
	poetry run black .
	poetry run isort .
