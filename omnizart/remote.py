"""Functions related to networking.

Containing functions to download files from the internet. Supports
download *zipped* files from Google Drive.

"""
# pylint: disable=R0914,R0915,W0612
import os
import sys
import time
import zipfile
import urllib.request
import http.cookiejar

from omnizart.utils import ensure_path_exists


#: Mapping bytes to human-readable size unit.
SIZE_MAPPING = [(1, "B"), (2**10, "KB"), (2**20, "MB"), (2**30, "GB"), (2**40, "TB")]


class GDFileAccessLimited(Exception):
    """Custom exception on failing to download GD file.

    This exception is raised when the GD file is overly accessed during a certain
    period.
    """
    pass


def format_byte(size, digit=2):
    """Format the given byte size into human-readable string."""
    rounding = f".{digit}f"
    for idx, (bound, unit) in enumerate(SIZE_MAPPING):
        if size <= bound:
            bound, unit = SIZE_MAPPING[idx - 1]
            return f"{size/bound:{rounding}}{unit}"
    return str(size)


def download(url, file_length=None, save_path="./", save_name=None, cookie_file=None, unzip=False):
    """Download file from the internet.

    Download file from the remote URL, with progress visualization and dynamic downloading
    rate adjustment. Uses pure python built-in packages, no additional package requirement.

    Parameters
    ----------
    url: URL
        The file download url.
    file_length: float
        In bytes. If the length can't be retrieved from the response header, but can be
        obtained by other approaches, you can explicitly specify the length for progress
        visualization.
    save_path: Path
        The path to store the donwloaded file.
    save_name: str
        Explicitly specify the file name to be stored. If not given, default to parse the
        name from the given url.
    cookie_file: Path
        Path to the cookie file. Suitable for stateful download (e.g. Google Drive).
    unzip: bool
        Whether to unzip (decompress) the downloaded file (assumed zipped). Will not delete
        the original downloaded file.

    Returns
    -------
    path: Path
        The absolute path to the downloaded/extracted folder/file.
    """
    filename = os.path.basename(url) if save_name is None else save_name
    out_path = os.path.join(save_path, filename)
    ensure_path_exists(os.path.dirname(out_path))
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

    init_chunk_size_mb = 0.1
    chunk_size = round(init_chunk_size_mb * 2**20)
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
            speed_history = speed_history[-40:]  # Keep only 40 records
            avg_speed = sum(speed_history) / len(speed_history)
            start_t = time.time()
        sys.stdout.write('\033[2K\033[1G')
        print(f"Progress: 100%, {format_byte(total_size)}, {format_byte(avg_speed)}/s")

    unzip_done = True
    if unzip:
        print("Extracting files...")
        try:
            with zipfile.ZipFile(out_path) as zip_ref:
                members = zip_ref.infolist()
                for idx, member in enumerate(members):
                    percent_finished = (idx+1) / len(members)*100  # noqa: E226
                    sys.stdout.write('\033[2K\033[1G')
                    print(f"Progress: {percent_finished:.2f}% - {member.filename}", end="\r")
                    zip_ref.extract(member, path=save_path)
                print("")

                # Assert the first item name is the root folder's name.
                extracted_name = zip_ref.namelist()[0]
                assert extracted_name.endswith("/") or extracted_name.endswith("\\")
            return os.path.abspath(os.path.join(save_path, extracted_name)), unzip_done
        except zipfile.BadZipFile:
            print("File is not a zip file, do nothing...")
            unzip_done = False

    return os.path.abspath(out_path), unzip_done


def download_large_file_from_google_drive(url, file_length=None, save_path="./", save_name=None, unzip=False):
    """Google Drive file downloader.

    Download function dedicated for Google Drive files. Mainly to deal with download
    large files and the confirmation page.

    Parameters
    ----------
    url: URL
        Could be a full google drive download url or the file ID.
    file_length: float
        In bytes. If the length can't be retrieved from the response header, but can be
        obtained by other approaches, you can explicitly specify the length for progress
        visualization.
    save_path: Path
        Path to store the downloaded file.
    save_name: str
        Explicitly specify the file name to be stored. If not given, default to parse the
        name from the given url.
    unzip: bool
        Whether to unzip (decompress) the downloaded file (assumed zipped). Will not delete
        the original downloaded file.

    Returns
    -------
    path: Path
        The absolute path to the downloaded/extracted folder/file.
    """
    if not (url.startswith("https://") or url.startswith("http://")):
        # The given 'url' is actually a file ID.
        assert len(url) == 33
        fid = url  # noqa: F841
        url = f"https://drive.google.com/uc?export=download&id={url}"
    else:
        id_start = url.find("id=") + 3
        fid = url[id_start:id_start+33]  # noqa: E226,F841

    cookie_jar = http.cookiejar.MozillaCookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    resp = opener.open(url)
    cookie_jar.save("./.cookie")
    cookie = resp.getheader("Set-Cookie")
    if cookie is None:
        # Actually a small file, without download confirmation.
        return download(url, file_length=file_length, save_path=save_path, save_name=save_name, unzip=unzip)

    if file_length is None:
        # Parse the file size from the returned page content.
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
                file_length = float(file_size[:idx]) * val
                break

    cols = cookie.split("; ")

    try:
        warn_col = [col for col in cols if "download_warning" in col][0]
    except IndexError:
        raise GDFileAccessLimited("The resource is temporarily unavailable due to file being overly accessed")

    confirm_id = warn_col.split("=")[1]
    url = f"{url}&confirm={confirm_id}"
    return download(
        url, file_length=file_length, save_path=save_path, save_name=save_name, cookie_file="./.cookie", unzip=unzip
    )
