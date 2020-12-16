import socket

hostname = '127.16.55.120'
port = 4502
addr = (hostname, port)

srv = socket.socket()
srv.bind(addr)
srv.listen(8)
print("waitting connect")

gpu_0 = open("GPU_0.txt", mode='a+')
gpu_1 = open("GPU_1.txt", mode='a+')
gpu_2 = open("GPU_2.txt", mode='a+')
gpu_3 = open("GPU_3.txt", mode='a+')
gpu_4 = open("GPU_4.txt", mode='a+')
gpu_5 = open("GPU_5.txt", mode='a+')
gpu_6 = open("GPU_6.txt", mode='a+')
gpu_7 = open("GPU_7.txt", mode='a+')

gpu_0.write('1\n')
gpu_1.write('1\n')
gpu_2.write('1\n')
gpu_3.write('1\n')
gpu_4.write('1\n')
gpu_5.write('1\n')
gpu_6.write('1\n')
gpu_7.write('1\n')

gpu_0.flush()
gpu_1.flush()
gpu_2.flush()
gpu_3.flush()
gpu_4.flush()
gpu_5.flush()
gpu_6.flush()
gpu_7.flush()

while True:
    connect_socket, client_addr = srv.accept()
    data = connect_socket.recv(1024)
    date = '%s'%(data.decode('utf-8'))
    print(data)

    if data == b'GPU:0; State:STOP':
        gpu_0.write('0\n') 
    if data == b'GPU:0; State:RUN':
        gpu_0.write('1\n') 

    if data == b'GPU:1; State:STOP':
        gpu_1.write('0\n') 
    if data == b'GPU:1; State:RUN':
        gpu_1.write('1\n') 

    if data == b'GPU:2; State:STOP':
        gpu_2.write('0\n') 
    if data == b'GPU:2; State:RUN':
        gpu_2.write('1\n') 

    if data == b'GPU:3; State:STOP':
        gpu_3.write('0\n') 
    if data == b'GPU:3; State:RUN':
        gpu_3.write('1\n') 

    if data == b'GPU:4; State:STOP':
        gpu_4.write('0\n') 
    if data == b'GPU:4; State:RUN':
        gpu_4.write('1\n') 

    if data == b'GPU:5; State:STOP':
        gpu_5.write('0\n') 
    if data == b'GPU:5; State:RUN':
        gpu_5.write('1\n') 

    if data == b'GPU:6; State:STOP':
        gpu_6.write('0\n') 
    if data == b'GPU:6; State:RUN':
        gpu_6.write('1\n') 

    if data == b'GPU:7; State:STOP':
        gpu_7.write('0\n') 
    if data == b'GPU:7; State:RUN':
        gpu_7.write('1\n') 

    gpu_0.flush()
    gpu_1.flush()
    gpu_2.flush()
    gpu_3.flush()
    gpu_4.flush()
    gpu_5.flush()
    gpu_6.flush()
    gpu_7.flush()

connect_socket.close()