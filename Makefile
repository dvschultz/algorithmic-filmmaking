.PHONY: run watch test

run:
	python main.py

watch:
	watchmedo auto-restart --patterns="*.py" --recursive --directory="./ui" --directory="./core" --directory="./models" -- python main.py

test:
	pytest tests/ -v
