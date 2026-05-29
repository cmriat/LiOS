set dotenv-load := true

dirpath := justfile_directory()

default:
    @just --choose

sync target="cc.coder":
    rsync --delete -avP "{{ dirpath }}/" {{ target }}:$(basename "{{ dirpath }}")/ --exclude-from=.rsync_exclude
