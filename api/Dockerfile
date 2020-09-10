# Dockerfile - this is a comment. Delete me if you want.
FROM python:3.7
COPY . /api
WORKDIR /api
RUN pip install --upgrade pip
RUN pip install --use-feature=2020-resolver -r requirements.txt
ENV FLASK_APP=api.py
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "8080"]