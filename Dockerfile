FROM python:3.10.9

WORKDIR /app/

COPY ./requirements.txt ./

RUN pip install -r requirements.txt

COPY ./model.py ./
COPY ./server.py ./
COPY ./prediction.py ./

EXPOSE 4000

CMD ["python","server.py"]