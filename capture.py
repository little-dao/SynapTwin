from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

def handle_all(address, *args):
    print(f"{address}: {args}")

dispatcher = Dispatcher()
dispatcher.set_default_handler(handle_all)  # <-- Catch all messages, no filtering

ip = "127.0.0.1"
port = 12345
print(f"Listening on {ip}:{port}")
server = BlockingOSCUDPServer((ip, port), dispatcher)
server.serve_forever()

# tell the person if they are stressed or not and what they can do to help