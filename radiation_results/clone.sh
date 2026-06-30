#!/usr/bin/env bash

PREFIX="$1"
BASE="$2"

shift 2

for level in "$@"; do
    cp "${PREFIX}.${BASE}.session" "${PREFIX}.${level}.session"

    sed -i \
        "s#/data/l${BASE}/#/data/l${level}/#g" \
        "${PREFIX}.${level}.session"
done
