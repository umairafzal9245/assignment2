install: 
		pip install --upgrade pip && \
		pip install -r requirements.txt

trainmodel:
		python model.py --ticker SILK
