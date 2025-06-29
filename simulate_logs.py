import json
import random
from datetime import datetime, timedelta

# Define the columns used in your dataset (excluding 'attack' and 'level')
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'difficulty_level'
]

# Some possible values for categorical fields (based on NSL-KDD)
protocol_types = ['tcp', 'udp', 'icmp']
services = ['http', 'ftp', 'smtp', 'domain_u', 'eco_i']
flags = ['SF', 'S0', 'REJ', 'RSTR']

# Generate random log entry
def generate_random_log(node_id, timestamp):
    return {
        'node_id': node_id,
        'timestamp': timestamp.isoformat(),
        'duration': random.randint(0, 10000),
        'protocol_type': random.choice(protocol_types),
        'service': random.choice(services),
        'flag': random.choice(flags),
        'src_bytes': random.randint(0, 100000),
        'dst_bytes': random.randint(0, 100000),
        'land': random.randint(0, 1),
        'wrong_fragment': random.randint(0, 3),
        'urgent': random.randint(0, 3),
        'hot': random.randint(0, 5),
        'num_failed_logins': random.randint(0, 5),
        'logged_in': random.randint(0, 1),
        'num_compromised': random.randint(0, 10),
        'root_shell': random.randint(0, 1),
        'su_attempted': random.randint(0, 1),
        'num_root': random.randint(0, 10),
        'num_file_creations': random.randint(0, 5),
        'num_shells': random.randint(0, 2),
        'num_access_files': random.randint(0, 5),
        'num_outbound_cmds': 0,
        'is_host_login': random.randint(0, 1),
        'is_guest_login': random.randint(0, 1),
        'count': random.randint(0, 500),
        'srv_count': random.randint(0, 500),
        'serror_rate': round(random.uniform(0, 1), 2),
        'srv_serror_rate': round(random.uniform(0, 1), 2),
        'rerror_rate': round(random.uniform(0, 1), 2),
        'srv_rerror_rate': round(random.uniform(0, 1), 2),
        'same_srv_rate': round(random.uniform(0, 1), 2),
        'diff_srv_rate': round(random.uniform(0, 1), 2),
        'srv_diff_host_rate': round(random.uniform(0, 1), 2),
        'dst_host_count': random.randint(0, 255),
        'dst_host_srv_count': random.randint(0, 255),
        'dst_host_same_srv_rate': round(random.uniform(0, 1), 2),
        'dst_host_diff_srv_rate': round(random.uniform(0, 1), 2),
        'dst_host_same_src_port_rate': round(random.uniform(0, 1), 2),
        'dst_host_srv_diff_host_rate': round(random.uniform(0, 1), 2),
        'dst_host_serror_rate': round(random.uniform(0, 1), 2),
        'dst_host_srv_serror_rate': round(random.uniform(0, 1), 2),
        'dst_host_rerror_rate': round(random.uniform(0, 1), 2),
        'dst_host_srv_rerror_rate': round(random.uniform(0, 1), 2),
        'difficulty_level': round(random.uniform(0, 1), 2)
    }


jsonl_path = "network_logs.jsonl"
base_time = datetime.now()
log_interval = timedelta(seconds=0.2) 

with open(jsonl_path, "w") as f:
    log_count = 0
    for _ in range(400):
        for node_id in ["ESP32_001", "ESP32_002", "ESP32_003"]:
            current_time = base_time + (log_count * log_interval)
            log = generate_random_log(node_id, current_time)
            f.write(json.dumps(log) + "\n")
            log_count += 1
