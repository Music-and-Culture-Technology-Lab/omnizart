import os
import time
import urllib.request


SIZE_MAPPING = [(1, "B"), (2**10, "KB"), (2**20, "MB"), (2**30, "GB"), (2**40, "TB")]


def format_byte(size, digit=3):
    rounding = f".{digit}f"
    for idx, (bound, unit) in enumerate(SIZE_MAPPING):
        if size <= bound:
            bound, unit = SIZE_MAPPING[idx - 1]
            return f"{size/bound:{rounding}}{unit}"
    return str(size)


def download(url, save_path="./"):
    chunk_size_mb = 1
    chunk_size_kb = chunk_size_mb * 1024
    chunk_size = chunk_size_kb * 1024
    filename = os.path.basename(url)
    out_path = os.path.join(save_path, filename)
    print(f"Output path: {out_path}")
    with open(out_path, "wb") as out:
        total_size = 0
        resp = urllib.request.urlopen(url)
        length = int(resp.getheader("Content-Length", -1))
        print(f"Total size: {format_byte(length)}")

        diff_t = 1
        speed_history = [chunk_size]
        avg_speed = sum(speed_history) / len(speed_history)
        start_t = time.time()
        while True:
            percent = f"{(total_size/length*100):.2f}" if length > 0 else "?"
            print(f"Progress: {percent}%, {format_byte(total_size)}, "
                    f"{format_byte(avg_speed)}/s"+" "*6, end="\r")  # noqa: E127,E226
            data = resp.read(chunk_size)
            if not data:
                break
            size = out.write(data)
            total_size += size

            diff_t = time.time() - start_t
            if diff_t > 0.2:
                chunk_size /= 1.5
            else:
                chunk_size *= 1.1
            chunk_size = round(chunk_size)
            speed_history.append(chunk_size)
            speed_history = speed_history[-40:]
            avg_speed = sum(speed_history) / len(speed_history)
            start_t = time.time()
        print(f"Progress: 100%, {format_byte(total_size)}, {format_byte(avg_speed)}/s"+" "*6)  # noqa: E226


if __name__ == "__main__":
    url = "https://drive.google.com/uc?export=download&id=1GVqlEq6we0xS9DoPK3vxCqpF1ZymxuGb"
    url = "https://drive.google.com/u/1/uc?export=download&id=1sjv9mpLFSjFeJsr8vhtqp80DRnOO5ZYJ"
    download(url)