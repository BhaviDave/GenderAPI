FROM tensorflow/tensorflow


RUN pip install flask
RUN pip install requests
RUN pip install numpy
RUN pip install Pillow
COPY . /app
WORKDIR /app

# Expose the Flask port
EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["run.py"]