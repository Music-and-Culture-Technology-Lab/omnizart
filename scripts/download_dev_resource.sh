#!/usr/bin/env python3

import os

from omnizart.remote import download_large_file_from_google_drive


download_large_file_from_google_drive(
    "10i8z1zH60a2coKEst47lELdkvZUmgd1b",
    file_length=65078525,
    save_path="./tests",
    save_name='resource.zip',
    unzip=True
)

os.remove("./tests/resource.zip")
