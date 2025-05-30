import threading
import time
import random
import requests
from flask import Flask, request, jsonify

class Node:
    def __init__(self, name, ip, port, neighbors):
        self.name = name
        self.ip = ip
        self.port = port
        self.address = f"http://{self.ip}:{self.port}"
        self.app = Flask(self.name)
        self.known_nodes = set(neighbors)

        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "ok", "node": self.name, "address": self.address})

        @self.app.route('/register', methods=['POST'])
        def register_node():
            data = request.get_json()
            peer_address = data.get("address")
            if peer_address:
                self.known_nodes.add(peer_address)
                return jsonify({"status": "registered", "known_nodes": list(self.known_nodes)})
            else:
                return jsonify({"status": "error", "reason": "No address provided"}), 400

        @self.app.route('/nodes', methods=['GET'])
        def get_nodes():
            return jsonify(list(self.known_nodes))

    def start_server(self):
        threading.Thread(target=self.app.run, kwargs={'host': self.ip, 'port': self.port}, daemon=True).start()
        print(f"{self.name} started at {self.address}")

    def register_with_peers(self):
        for peer in list(self.known_nodes):
            try:
                url = f"{peer}/register"
                response = requests.post(url, json={"address": self.address}, timeout=3)
                if response.status_code == 200:
                    returned_nodes = response.json().get("known_nodes", [])
                    self.known_nodes.update(returned_nodes)
                    print(f"[{self.name}] 向 {peer} 注册成功，已知节点更新: {self.known_nodes}")
                else:
                    print(f"[{self.name}] 注册到 {peer} 失败，状态码: {response.status_code}")
            except Exception as e:
                print(f"[{self.name}] 注册到 {peer} 失败: {str(e)}")

    def sync_nodes(self):
        while True:
            time.sleep(10)
            for peer in list(self.known_nodes):
                if peer == self.address:
                    continue
                try:
                    url = f"{peer}/nodes"
                    response = requests.get(url, timeout=3)
                    if response.status_code == 200:
                        peer_nodes = response.json()
                        self.known_nodes.update(peer_nodes)
                        print(f"[{self.name}] 从 {peer} 同步成功，当前已知节点: {self.known_nodes}")
                except Exception as e:
                    print(f"[{self.name}] 同步节点失败 {peer}: {str(e)}")

# ⭐⭐⭐ 只这里：生成IP用你的规则 ⭐⭐⭐
def generate_unique_ips(num_nodes):
    existing_ips = set()
    ip_list = []
    while len(ip_list) < num_nodes:
        ip = f"192.168.1.{random.randint(1, 255)}"
        if ip not in existing_ips:
            existing_ips.add(ip)
            ip_list.append(ip)
    return ip_list

def main():
    num_nodes = 5
    neighbors_count = 2
    base_port = 5000

    ip_list = generate_unique_ips(num_nodes)
    node_addresses = [f"http://{ip}:{base_port + idx}" for idx, ip in enumerate(ip_list)]

    # 给每个节点分配 neighbors（简单点，不保证双向）
    node_neighbors = {}
    for i, my_ip in enumerate(ip_list):
        my_addr = f"http://{my_ip}:{base_port + i}"
        available_peers = [addr for addr in node_addresses if addr != my_addr]
        neighbors = random.sample(available_peers, min(neighbors_count, len(available_peers)))
        node_neighbors[my_addr] = neighbors

    nodes = []
    for idx, (addr, neighbors) in enumerate(node_neighbors.items()):
        ip = ip_list[idx]
        port = base_port + idx
        name = f"Node{idx+1}"
        node = Node(name, ip, port, neighbors)
        nodes.append(node)

    for node in nodes:
        node.start_server()

    time.sleep(2)  # 等服务器起来

    for node in nodes:
        node.register_with_peers()
        threading.Thread(target=node.sync_nodes, daemon=True).start()

    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()
