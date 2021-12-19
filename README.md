## requirements

```ps1
python -m venv venv

.\venv\scripts\activate

pip install -r requirements.txt
```

## start

```text
создать файл `.env`

записать в файл:

FLASK_ENV=development
FLASK_APP=app/main.py
```

```ps1
.\venv\scripts\activate

python -m flask run -p 9000
```
