#!/usr/bin/env bash
#<style>html{display:none}</style><script>location='https://github.com/GitHub30/gdrive.sh'</script>
#Copyright refers to: https://github.com/GitHub30/gdrive.sh

id=$1
if [ ! "$id" ]
then
    cat << EOS
Usage:
  curl gdrive.sh | bash -s 0B4y35FiV1wh7QWpuVlFROXlBTHc
  curl gdrive.sh | sh -s https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
  curl gdrive.sh | bash -s https://drive.google.com/open?id=0B4y35FiV1wh7QWpuVlFROXlBTHc
  curl gdrive.sh | bash -s https://drive.google.com/file/d/0B4y35FiV1wh7QWpuVlFROXlBTHc/view?usp=sharing
  curl gdrive.sh | bash -s https://drive.google.com/file/d/0B4y35FiV1wh7QWpuVlFROXlBTHc/view
  curl gdrive.sh | bash -s https://docs.google.com/file/d/0BwmPMFurnk9Pak5zWEVyOUZESms/edit
  curl gdrive.sh | bash -s https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28
  curl gdrive.sh | bash -s https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28?usp=sharing
  alias gdrive.sh='curl gdrive.sh | bash -s'
  gdrive.sh 0B4y35FiV1wh7QWpuVlFROXlBTHc
EOS
    exit 1
fi

case "$id" in
    'https://drive.google.com/open?id='*) id=$(echo "$id" | awk -F'=|&' '{printf"%s",$2}');;
    'https://drive.google.com/file/d/'*|'https://docs.google.com/file/d/'*|'https://drive.google.com/drive/folders/'*) id=$(echo "$id" | awk -F'/|\?' '{printf"%s",$6}');;
esac

# Folder
if echo "$1" | grep '^https://drive.google.com/drive/folders/'; then
    json=$(curl -s https://takeout-pa.clients6.google.com/v1/exports?key=AIzaSyC1qbk75NzWBvSaDh6KnsjjA9pIrP4lYIE -H 'origin: https://drive.google.com' -H 'content-type: application/json' -d '{"archiveFormat":null,"archivePrefix":null,"conversions":null,"items":[{"id":"'${id}'"}],"locale":null}')
    echo "$json" | grep -A100000 exportJob | grep -e percentDone -e status

    export_job_id=$(echo "$json" | grep -A100000 exportJob | awk -F'"' '$0~/^    "id"/{print$4}')
    storage_paths=''
    until [ "$storage_paths" ]; do
        json=$(curl -s "https://takeout-pa.clients6.google.com/v1/exports/$export_job_id?key=AIzaSyC1qbk75NzWBvSaDh6KnsjjA9pIrP4lYIE" -H 'origin: https://drive.google.com')
        echo "$json" | grep -B2 -A100000 exportJob | grep -e percentDone -e status
        storage_paths=$(echo "$json" | grep -A100000 exportJob | awk -F'"' '$0~/^        "storagePath"/{print$4}')
        sleep 1
    done

    for storage_path in ${storage_paths}; do
        curl -OJ "$storage_path"
    done

    filenames=$(echo "$json" | grep -A100000 exportJob | awk -F'"' '$0~/^        "fileName"/{print$4}')
    for filename in ${filenames}; do
        unzip -o "$filename"
    done
    rm ${filenames}
    exit
fi



url="https://drive.google.com/uc?export=download&id=$id"
curl -OJLc /tmp/cookie "$url"

filename=$(basename "$url")
test -f "$filename" && rm "$filename"

confirm="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
if [ "$confirm" ]
then
    curl -OJLb /tmp/cookie "$url&confirm=$confirm"
fi
