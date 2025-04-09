#!/bin/sh
mkdir test_assets
cd test_assets || exit
wget "https://github.com/vguzov/cloudrender/releases/download/v1.3.6/test_assets.zip" -O test_assets.zip
unzip test_assets.zip
rm test_assets.zip