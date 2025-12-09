FROM python:3.12-slim

ENV PYTHONUNBUFFERED 1
ENV APP_HOME /usr/src/app

WORKDIR $APP_HOME

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . $APP_HOME

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "main:app" ]