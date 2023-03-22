install: 
		pip install --upgrade pip && \
		pip install -r requirements.txt

format:
		black *.py
		
lint:
		pylint --disable=R,C *.py

trainmodel:
		python model.py --ticker UBER
