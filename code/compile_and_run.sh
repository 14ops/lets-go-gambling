#!/bin/bash

# Applied Probability and Automation Framework
# Java GUI Compilation and Execution Script

echo "Compiling Java GUI..."

# Create lib directory if it doesn't exist
mkdir -p lib

# Download Gson library if not present
if [ ! -f "lib/gson-2.10.1.jar" ]; then
    echo "Downloading Gson library..."
    wget -O lib/gson-2.10.1.jar https://repo1.maven.org/maven2/com/google/code/gson/gson/2.10.1/gson-2.10.1.jar
fi

# Compile Java source
javac -cp "lib/gson-2.10.1.jar" -d . src/GameControlPanel.java

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Running Java GUI..."
    java -cp ".:lib/gson-2.10.1.jar" GameControlPanel
else
    echo "Compilation failed!"
    exit 1
fi

