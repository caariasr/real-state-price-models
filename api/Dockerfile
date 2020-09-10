# Dockerfile - this is a comment. Delete me if you want.
FROM python:3.7
COPY . /api
WORKDIR /api
RUN pip install --upgrade pip
RUN pip install --use-feature=2020-resolver -r requirements.txt
ENTRYPOINT ["python"]
CMD ["api.py"]