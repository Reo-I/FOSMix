#!/bin/bash

# Define the package names to be processed in an array
PACKAGE_NAMES=("segmentation-models-pytorch" "torchvision")

# Perform the process for each package
for PACKAGE_NAME in "${PACKAGE_NAMES[@]}"; do
  # Generate the directory name from the package name (replace hyphen with underscore)
  DIRECTORY_NAME=${PACKAGE_NAME//-/_}

  # Use the pip show command to get the installation location of the package
  PACKAGE_LOCATION=$(pip show $PACKAGE_NAME | grep Location | cut -d ' ' -f 2)

  # Get the directory of the package
  PACKAGE_DIR=$PACKAGE_LOCATION/$DIRECTORY_NAME

  # Search all .py files in the package directory and replace nn.ReLU(inplace=True) with nn.ReLU()
  find $PACKAGE_DIR -name "*.py" -type f -exec sed -i '' 's/nn.ReLU(inplace=True)/nn.ReLU()/g' {} \;
done
