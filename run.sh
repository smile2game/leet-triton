rm trace/*
nsys profile --trace=cuda,nvtx,osrt -o nsys_trace python matmul.py
# python matmul.py