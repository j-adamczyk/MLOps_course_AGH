# Laboratory 5 instruction part 1 - DVC

## DVC introduction

[Data Version Control (DVC)](https://dvc.org/) is the most popular solution for
implementing data versioning. It's tightly coupled with Git, but allows efficient
storage of large data and model files. It uses separate remote data storage for
handling those large files, and only stores a small metadata file in your Git
repository.

DVC also has other features, e.g. data pipelines and experiment versioning, but
we will focus here on the "Git for data" aspect here.

## Git configuration

To use DVC, you need a Git repository. Overall, DVC is very tightly integrated with
Git, and requires a mix of Git and DVC commands. The latter are also very strongly
inspired by typical Git commands.

Create a repository on GitHub, commit, and push the provided laboratory files there.
Using CLI is recommended, since you will need it for DVC commands:
- add all files to local staging area: `git add .`
- commit them with a given message: `git commit -m "initial commit"`
- push changes to remote repository: `git push`

Note that we have a `.gitignore` file - it will exclude certain files and directories
from being pushed to Git repository. It's always useful to use it, but particularly
in MLOps, where we work with large files, e.g. datasets and models. DVC will additionally
create its own `.gitignore` files to avoid pushing large files to Git.

**Warning:** we do not explicitly mention any commit names in further instructions, but
make sure that your commit messages are readable and your Git history is reasonably clean.
Rename or squash commits if necessary before submitting the lab.

## First DVC steps

DVC is initialized in the existing Git repository by running `dvc init`, do it now.
Message printed in the CLI should look like:
```
Initialized DVC repository.

You can now commit the changes to git.
```

This creates:
- `.dvcignore` file, working similarly to `.gitignore`, e.g. for ignoring temporary
  data files
- `.dvc` directory, managing DVC state

Those file **need** to be commited to your Git repository. They are responsible for
tracking the Git-data relationship.

Let's manage some data then! We will use the [Ames housing](https://www.openintro.org/book/statdata/?data=ames)
dataset about house prices in 2006-2010 in Ames, Iowa. However, since we have the dates
available, we have prepared a realistic data split:
- data from years 2006-2008 (inclusive) is "currently available"
- data from 2009-2010 will be gathered in the future (as part of homework)

For now, download just the former one [from the Google Drive](https://drive.google.com/file/d/1_Y0LPWOTU3VnaS1t0HMDa9_G2MWtWBRp/view?usp=sharing),
and put it in a `data` directory. You can also run:
```commandline
mkdir -p data
```
```commandline
wget \
    --no-check-certificate \
    "https://drive.google.com/uc?export=download&id=1_Y0LPWOTU3VnaS1t0HMDa9_G2MWtWBRp" \
    -O data/ames_data_2006_2008.parquet
```

Inspect the file by using the `ames_inspect_data.py` script. It is implemented with
[click framework](https://click.palletsprojects.com/en/stable/), which is quite powerful
and easy to use for implementing CLI applications.

Now, let's start tracking that file by adding it to DVC:
```commandline
dvc add data/ames_data_2006_2008.parquet
```

This created the `data/ames_data_2006_2008.parquet.dvc` file, with contents like:
```
outs:
- md5: 7f045b7f24d1af6daf02a075a188432d
  size: 186512
  hash: md5
  path: ames_data_2006_2008.parquet
```
This way, DVC tracks the MD5 hash of the dataset, which allows for efficient checks
if it has changed and requires update during subsequent operations.

Also, `data/.gitignore` was created. It removes the `.parquet` file from Git, as we
want it to be tracked remotely only by DVC.

Those DVC files need to be pushed to the Git repository, synchronizing DVC data state
with a Git commit:
```commandline
git add .dvc data
```

Notice that the actual dataset was not added. Make the commit and push it to the remote.

## Configuring DVC remotes

Currently, Git ignores our data, and DVC tracks it locally. Now we need to create the
**remote data repository**, or **remote** for short, an analogue of a GitHub repository.
DVC supports [many different storage types](https://dvc.org/doc/user-guide/data-management/remote-storage#supported-storage-types),
e.g. AWS S3, MinIO, GCP, anything available over SSH or SCP, or even just Google Drive.
For simplicity, here we will configure a "local remote", i.e. just a separate directory
on the same machine. This can be useful when working on large servers and big projects,
where data is kept local for efficiency or regulatory reasons.

To add a remote, `dvc remote add REMOTE_NAME REMOTE_LOCATION` command is used, typically
with `--default` (set as default) option:
```commandline
mkdir -p remote_data
```
```commandline
dvc remote add --default local_remote remote_data
```

If you provide e.g. `s3://bucket_name/dvc_prefix`, DVC will automatically recognize the
remote as AWS S3, and try to set that up.

To push the data to the remote storage, run:
```commandline
dvc push
```

Inspect the `remote_data` directory now. You can see the file there, identified by the
MD5 checksum of its contents.

## Exercise 1

1. Make a Git commit and push it to remote. The `.dvc/config` file has changed, since
   we added a new DVC remote, and we need to track that.
2. Download the [ames_description.txt file from Google Drive](https://drive.google.com/file/d/1wJkhdOAkYAiZwqDbDevFdSvkhxh8mpNS/view?usp=drive_link)
   and put it in the `data` directory. This file contains descriptions of data features
   and their possible values.
3. Add the new file to DVC, and add the resulting `.dvc` file to Git.
4. Push the file to DVC remote repository.
5. Make a new Git commit, updating the tracked DVC files status.

## Adding new dataset version

Now we have the raw dataset added, along with its description. Let's assume that we now
want to work on the data cleaning, and implemented the `ames_data_cleaning.py` script.
It's a commandline Python script, taking `--file-path` argument, cleaning and transforming
the raw features. It will replace the input file with the cleaned one. This is a typical
operation if you have big data, and cannot really afford to keep the data in all stages
of transformation. Educationally, it will nicely showcase the usefulness of the DVC
versioning.

## Exercise 2

1. Run the script, replacing the raw data with new, cleaned file.
2. Inspect the resulting file with `ames_inspect_data.py` script, to ensure that it really
   changed as expected.
3. Add it to DVC, just like you would a new file, to update the data.
4. Add the resulting `.dvc` file to Git.
5. Push files to DVC and Git repositories.

## Changing dataset versions

Now that we have quite a few commits and data files, we can actually use the versioning
in practice and go back in time, to the previous dataset version.

First experiment - delete the newly replaces dataset file, and then perform `dvc checkout`.
Since you have its `.dvc` file, DVC knows what file should be there, and will pull it for you.

Now, go back in Git commits before you replaces the dataset with the cleaned data. To go back
`N` commits, run `git checkout HEAD~N` (replace `N` with however many commits you need). Then
run `dvc checkout`. It will ensure the same data state as you had at the previous commit,
without applied changes. You should see commandline output somewhat like this:
```
Building workspace index
Comparing indexes
Applying changes
M       data/ames_data_2006_2008.parquet
```

Inspect the file by using `ames_inspect_data.py` script, and verify that we indeed went back
to the original data.

This is the basic DVC workflow! Versioning also works great with branches, pull requests, and
other remotes. However, the basic principle is the same as here - Git tracks the code, and DVC
tracks heavy files like data and models. If you go back in Git history, you can always run
`dvc checkout` and get exactly the same data that you had at the time of that commit. At the
same time, DVC is slightly decoupled from Git, so you can update code without touching the data,
giving you the necessary flexibility.

You can automate many of those parts if you want by using
[dvc install](https://dvc.org/doc/command-reference/install), which integrate DVC tightly with
Git by using pre-commit hooks. However, this automation does not give you full control, so there
is a tradeoff there.
