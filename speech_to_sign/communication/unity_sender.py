import socket

HOST = "127.0.0.1"
PORT = 5050


def send_to_unity(word):
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((HOST, PORT))

        client.send(word.encode())

        client.close()

        print("Sent to Unity:", word)

    except:
        print("Unity not connected")