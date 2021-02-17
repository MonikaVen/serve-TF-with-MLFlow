FROM tensorflow/tensorflow
RUN apt update && apt upgrade -y
RUN apt install python3 python3-pip
WORKDIR /application/
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY templates/ templates/
RUN mkdir static
RUN mkdir logs
COPY app.py .
CMD python3 app.py 
EXPOSE 5000