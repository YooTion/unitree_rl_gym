#!/usr/bin/env python3
import socket
import json
import time

HOST = '192.168.1.105'
PORT = 9000

def run():
    while True:
        try:
            with socket.create_connection((HOST, PORT), timeout=2) as s:
                print('[Client] 已连接服务器')
                buf = ''
                while True:
                    data = s.recv(1024).decode('utf-8')
                    if not data:
                        break
                    buf += data
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        if line:
                            obj = json.loads(line)
                            print(f"yaw = {obj['yaw']:7.3f} rad")
        except (socket.error, ConnectionRefusedError):
            print('[Client] 连接失败，3 秒后重试...')
            time.sleep(3)

if __name__ == '__main__':
    run()