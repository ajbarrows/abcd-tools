# Download

## abcd_tools.download.NDAFastTrack

A collection of tools to make downloading a subset of ABCD Stuy FastTrack imaging data possible (for mortals).

In general, this is a wrapper around (https://github.com/NDAR/nda-tools)[`nda-tools`] which is currently used as a CLI tool. As such, `abcd_tools.download.NDAFastTrack` calls `downloadcmd` as a subprocess. Future versions may interact directly to reduce reliance on presumed operating system behavior.

**Important** You will need to:
- (https://nda.nih.gov/nda/access-data-info)[Have an account] with the NIMH Data Archive (NDA)
- (https://nda.nih.gov/ndarpublicweb/Documents/Accessing+Shared+Data+Sept_2021-1.pdf)[Create a data package] for the FastTrack data *making sure that associated files are included*
    - *Note:* A guide is forthcoming.
    - *Also Note:* This package is only tested with task fMRI data, but it may work with other types.


Recommended: Configure `nda-tools` to use your credentials systemwide:

```bash
pip install nda-tools
```

```bash
import keyring
keyring.set_password('nda-tools', 'YOUR_NDA_USERNAME', 'NEW_PASSWORD')
```

### Useage

`nda-tools` uses a "manifest" file to assemble AWS S3 links for the files you request through your data package. In order to parse that file to supply `nda-tools` with a list of S3 links we **wish** to download, we must first obtain the manifest file:

```python
from abcd_tools.download.NDAFastTrack import ManifestDownloader, ManifestParser

data_package = "1234567"
username = "YOUR_NDA_USERNAME"

manifest_downloader = ManifestDownloader(dp=data_package, username=username)
manifest_downloader.download()
```

*Note:* It is not currently possible to choose the download location for this file using this interface. `nda-tools` uses as default download location `~/NDA/nda-tools/downloadcmd/packages`.


Now we can explore available subject IDs, time points, and tasks:

```python
parser = ManifestParser(dp=data_package)
parser.parse()
```

`parser.metadata` will return a Pandas DataFrame which the user can query to for available data. Currently, `parser.s3_links` will return *all* available S3 links in your data package. This is rarely desirable behavior.

To obtain specific download paths:

```python
subjects=['NDARINVxxxxxxxx']
tasks=['MID']
timepoints=['baselineYear1Arm1']

parser = ManifestParser(dp=data_package, subjects=subjects, tasks=tasks, timepoints=timepoints)
parser.parse()
```

`s3_links` now contains only those download locations which match our query (i.e., all baseline 'MID' imaging files for participant 'NDARINVxxxxxxxx'; this should, at most, be two links)

Now, write these links to disk in a loacation accessible to `nda-tools`:

```python
parser.write_s3_links(fname="s3_links.csv")
```

Then, to actually download the associated files:

```python
from abcd_tools.download.NDAFastTrack import Downloader

download_dir = "./data"
downloader = Downloader(dp=data_package, username=username, download_directory=download_dir)
downloader.download("s3_links.csv")
```

Each downloaded file represents one imaging run. To be fully (https://bids.neuroimaging.io)[BIDS-compliant], all images acquired during a single session should live in the same directory. Extracting the compressed files using `tarfile` accomplishes this:

```python
import os
from abcd_tools.download.NDAFastTrack import FastTrackReorganizer

source = os.path.join(download_dir, 'fmriresults01')
target = "./out"

reorganizer = FastTrackReorganizer(source=source, target=target)
reorganizer.reorganize()
```

*Note:* The resulting files in `./out` will be uncompressed and effectively duplicates of the source files, which may cause storage concerns.
