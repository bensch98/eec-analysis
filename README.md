# Analysis of Electrical and Electronic Components Dataset

This repository can recreate all statistics mentioned in the corresponding paper.

## Files

- `features.py`: Computes statistics on for the separate features and feature instances.
- `geoemtry.py`: Computations related to geometry. HKS implementation stems from [nmwsharp/diffusion-net](https://github.com/nmwsharp/diffusion-net)
- `stats.py`: Script to reproduce the statistics and save them into `pandas.DataFrame`.
- `utils.py`: Helper / utility functions
- `requirements.txt`: Dependencies used in the python scripts.

## Metadata

The metadata of the files that were used are in `parts.json`. For each component following fields were stored. It's a doubly nested dictionary. The first key is the component id, while the second key is one of the metadata fields.

Fields            | Description
----------------- | -----------
Name              | Product name of manufacturer
Manufacturer      | Name of Manufacturer
Part Type         | Manufacturer specific type to group components
Technology        | 1st level filter option
Category          | 2nd level filter option
Subcategory       | 3rd level filter option
Series            | Manufacturer specific description for a series of products
Width             | Width of component in mm
Height            | Height of component in mm
Length            | Length of component in mm
Voltage           | Voltage in Volt
Current           | Current in Ampere
Power             | Power in kW
Part Status       | Whether it was available when the CAD file was downloaded
Part Description  | Description of the component
External Document | URL to the acutal product page of the manufacturer, where the STEP file can be downloaded and further information can be scraped
sha256            | sha256 hash to check for geometric unique STEP files
