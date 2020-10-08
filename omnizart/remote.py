import os
import time
import urllib.request
import http.cookiejar


SIZE_MAPPING = [(1, "B"), (2**10, "KB"), (2**20, "MB"), (2**30, "GB"), (2**40, "TB")]


def format_byte(size, digit=3):
    rounding = f".{digit}f"
    for idx, (bound, unit) in enumerate(SIZE_MAPPING):
        if size <= bound:
            bound, unit = SIZE_MAPPING[idx - 1]
            return f"{size/bound:{rounding}}{unit}"
    return str(size)


def download(url, file_length=None, save_path="./", save_name=None, cookie_file=None):
    filename = os.path.basename(url) if save_name is None else save_name
    out_path = os.path.join(save_path, filename)
    print(f"Output path: {out_path}")

    total_size = 0
    if cookie_file is None:
        resp = urllib.request.urlopen(url)
    else:
        cookie_jar = http.cookiejar.MozillaCookieJar()
        cookie_jar.load(cookie_file)
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
        resp = opener.open(url)

    length = int(resp.getheader("Content-Length", -1))
    if length == -1 and file_length is not None:
        length = file_length
    print(f"Total size: {format_byte(length)}")

    init_chunk_size_mb = 1
    chunk_size = init_chunk_size_mb * 2**20
    diff_t = 1
    speed_history = [chunk_size]
    avg_speed = sum(speed_history) / len(speed_history)
    with open(out_path, "wb") as out:
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


def download_large_file_from_google_drive(url, save_path="./", save_name=None):
    if not url.startswith("https://"):
        # The given 'url' is actually a file ID.
        assert len(url) == 33
        fid = url
        url = f"https://drive.google.com/uc?export=download&id={url}"
    else:
        id_start = url.find("id=") + 3
        fid = url[id_start:id_start+33]  # noqa: E226

    cookie_jar = http.cookiejar.MozillaCookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    resp = opener.open(url)
    cookie_jar.save("./.cookie")
    cookie = resp.getheader("Set-Cookie")
    if cookie is None:
        # Actually a small file, without download confirmation.
        download(url, save_path=save_path)
        return

    page = []
    while True:
        data = resp.read(2**15)
        if not data:
            break
        page.append(data.decode("UTF-8"))
    page = "".join(page)
    hack_idx_start = page.find("(") + 1
    hack_idx_end = page.find(")")
    file_size = page[hack_idx_start:hack_idx_end]

    # Parse file size as byte
    mapping = {"M": 2**20, "G": 2**30}
    for key, val in mapping.items():
        idx = file_size.find(key)
        if idx != -1:
            size = float(file_size[:idx]) * val
            break

    cols = cookie.split("; ")
    warn_col = [col for col in cols if "download_warning" in col][0]
    confirm_id = warn_col.split("=")[1]
    url = f"{url}&confirm={confirm_id}"
    download(url, file_length=size, save_path=save_path, save_name=save_name, cookie_file="./.cookie")


if __name__ == "__main__":
    # URL = "https://drive.google.com/uc?export=download&id=1GVqlEq6we0xS9DoPK3vxCqpF1ZymxuGb"
    URL = "https://drive.google.com/uc?export=download&id=1sjv9mpLFSjFeJsr8vhtqp80DRnOO5ZYJ"
    download(URL)
    download_large_file_from_google_drive(URL)
