import socket

def client_send(device, state):
    hostname =  '127.16.55.120' #'172.17.0.2'
    port = 4502
    addr = (hostname,port)

    clientsock = socket.socket()
    clientsock.connect(addr)

    if state == 0:
        say = 'GPU:{}; State:STOP'.format(device)
    if state == 1:
        say = 'GPU:{}; State:RUN'.format(device)
    clientsock.send(say.encode('utf-8'))
    clientsock.close()

if __name__ =='__main__':
    client_send(7, 1)