from mpi4py import MPI
import pickle
import heapq
from collections import defaultdict

TOP_N = 3
DATA_FILE = "traffic_data.txt"

def parse_line(line):
    timestamp, light_id, car_count = line.strip().split(",")
    return timestamp.strip(), light_id.strip(), int(car_count.strip())

def process_data(lines):
    local_counts = defaultdict(int)
    for line in lines:
        _, light_id, cars = parse_line(line)
        local_counts[light_id] += cars
    return dict(local_counts)

def merge_counts(counts_list):
    merged = defaultdict(int)
    for counts in counts_list:
        for light_id, count in counts.items():
            merged[light_id] += count
    return dict(merged)

def show_top_n(counts, n=TOP_N):
    top_n = heapq.nlargest(n, counts.items(), key=lambda x: x[1])
    print(f"\nTop {n} Most Congested Traffic Lights:")
    for light_id, total_cars in top_n:
        print(f"Traffic Light: {light_id}, Cars Passed: {total_cars}")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Master process
        with open(DATA_FILE, 'r') as f:
            lines = f.readlines()

        chunk_size = len(lines) // (size - 1)
        for i in range(1, size):
            chunk = lines[(i - 1) * chunk_size : i * chunk_size] if i != size - 1 else lines[(i - 1) * chunk_size:]
            comm.send(chunk, dest=i, tag=11)

        results = []
        for i in range(1, size):
            data = comm.recv(source=i, tag=22)
            results.append(pickle.loads(data))

        final_counts = merge_counts(results)
        show_top_n(final_counts)

    else:
        # Worker processes
        received_data = comm.recv(source=0, tag=11)
        local_result = process_data(received_data)
        comm.send(pickle.dumps(local_result), dest=0, tag=22)

if __name__ == "__main__":
    main()
