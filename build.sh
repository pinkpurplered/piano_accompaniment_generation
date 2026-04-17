#!/bin/bash
set -e

echo "Building frontend..."
cd front-end
npm install
npm run build
cd ..

echo "Copying build to backend..."
rm -rf back-end/static
cp -r front-end/build back-end/static

# Ensure midi directory exists
mkdir -p back-end/static/midi

echo "Done! The production UI is now available in back-end/static/"
