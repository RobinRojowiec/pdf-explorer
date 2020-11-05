#!/bin/bash

# build latest images
docker build -t pdf-explorer .

# kill and rm old container
docker kill pdf_explorer && docker rm pdf_explorer

# start new container
docker run -p 0.0.0.0:8501:8501 --name pdf_explorer pdf-explorer:latest