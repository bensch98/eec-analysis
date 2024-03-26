# Analysis of Electrical and Electronic Components Dataset

This repository can recreate all statistics mentioned in the corresponding paper.
Link to open-access dataset: [Electrical and Electronic Components Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/D3ODGT)

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

## Citation

To cite the data descriptor in `Nature Scientific Data` use:
```bibtex
@article{Scheffler2024,
  author    = {Scheffler, Benedikt and Br\"{u}ndl, Patrick and Nguyen, Huong Giang and Stoidner, Micha and Franke, J\"{o}rg},
  title     = {A Dataset of Electrical Components for Mesh Segmentation and Computational Geometry Research},
  journal   = {Scientific Data},
  volume    = {11},
  number    = {1},
  pages     = {309},
  year      = {2024},
  doi       = {10.1038/s41597-024-03155-w},
  url       = {https://doi.org/10.1038/s41597-024-03155-w},
  issn      = {2052-4463},
  date      = {2024/03/22},
}
```

To cite the data itself use:
```bibtex
@data{DVN/D3ODGT_2023,
  author    = {Scheffler, Benedikt and Br\"{u}ndl, Patrick},
  publisher = {Harvard Dataverse},
  title     = {{Electrical and Electronic Components Dataset}},
  year      = {2023},
  version   = {V1},
  doi       = {10.7910/DVN/D3ODGT},
  url       = {https://doi.org/10.7910/DVN/D3ODGT}
}
```
