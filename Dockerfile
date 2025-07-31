FROM Python:3.10

COPY . /VVulDet

WORKDIR /VVulDet

RUN chmod +x ./run.sh
RUN ./run.sh

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN ./zrun/getjoern.sh

RUN ./zrun/dataprocessing.sh
RUN ./zrun/trainmain.sh


